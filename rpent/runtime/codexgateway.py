"""CodexGateway Chat client with an isolated, explicitly released tool session."""

from __future__ import annotations

import asyncio
import os
import queue
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from rpent.runtime.api import ApiAgentRuntime
from rpent.runtime.base import PlannerResult
from rpent.runtime.telemetry import RequestObserver
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger

logger = get_logger("codexgateway")

# The gateway discards a session whose turn failed upstream (a 502 or 504 in
# the middle of a tool turn). A tool exchange sent afterwards is refused in one
# of two ways, both tool_call_mismatch: the session id is unknown to the gateway
# ("No pending tool call", seen in the 1.2.0 D10 run of 2026-10-12, with no 5xx
# reaching the client), or a session exists without its thread ("cannot be
# replayed"). Either way the gateway no longer holds the conversation. Other
# tool-call mismatches are protocol errors and still fail.
_SESSION_LOST = (
    "Tool history cannot be replayed into a new session",
    "No pending tool call in this session",
)
_MAX_SESSION_RESTARTS = 2


def _session_lost(exc: Exception) -> bool:
    if not isinstance(exc, ModelHTTPError) or exc.status_code != 400:
        return False
    text = f"{exc.body} {exc}"
    return "tool_call_mismatch" in text and any(message in text for message in _SESSION_LOST)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "") for part in content or [] if part.get("type") == "text"
    )


# The gateway refuses prompts over 200,000 characters. One executor session
# spans a whole episode (1.0.10 D06: 117 tool results, about 1.46M characters
# of projected context), so a restart keeps the task and the latest exchanges.
_RESTART_HISTORY_CHARS = 150_000
_IMAGE_CHARS = 1_000


def _size(message: ChatCompletionMessageParam) -> int:
    content = message.get("content")
    if isinstance(content, str):
        return len(content)
    return sum(
        len(part.get("text", "")) if part.get("type") == "text" else _IMAGE_CHARS
        for part in content or []
    )


def _flatten_history(
    mapped: list[ChatCompletionMessageParam],
    budget: int = _RESTART_HISTORY_CHARS,
) -> list[ChatCompletionMessageParam]:
    """Plain messages a new gateway session accepts: no tool roles or calls.

    The calls already ran; their results are history the model must see, not
    tool output to replay, so nothing is executed again. Instructions, the task
    message and the most recent exchanges are kept within ``budget``
    characters; every tool result carries the current state.
    """
    flat: list[ChatCompletionMessageParam] = []
    for message in mapped:
        if message["role"] == "assistant":
            calls = [
                f"[called {call['function']['name']}({call['function']['arguments']})]"
                for call in message.get("tool_calls") or []
            ]
            text = "\n".join(filter(None, [_text(message.get("content")), *calls]))
            flat.append({"role": "assistant", "content": text or "[no text]"})
        elif message["role"] == "tool":
            flat.append({
                "role": "user",
                "content": f"[result of {message['tool_call_id']}]\n"
                + _text(message["content"]),
            })
        else:
            flat.append(message)
    head = [m for m in flat if m["role"] in {"system", "developer"}]
    rest = [m for m in flat if m["role"] not in {"system", "developer"}]
    first, later = rest[:1], rest[1:]
    used = sum(_size(m) for m in head + first)
    tail: list[ChatCompletionMessageParam] = []
    for message in reversed(later):
        if tail and used + _size(message) > budget:
            break
        tail.insert(0, message)
        used += _size(message)
    omitted = len(later) - len(tail)
    note = [{
        "role": "user",
        "content": f"[{omitted} earlier messages omitted after a gateway session "
        "loss; the latest tool results below carry the current state]",
    }] if omitted else []
    return head + first + note + tail


class _CodexGatewayChatModel(OpenAIChatModel):
    """Send the current exchange to a gateway-owned conversation.

    Pydantic keeps the complete history for local reasoning/audit consumers.
    Only its wire representation is reduced: the gateway already owns every
    previous exchange in this solve's isolated native thread. When the gateway
    has lost that thread, one request carries the whole history, flattened,
    into a new session.
    """

    captures_wire_images = True
    restart_session: Any = None
    _flatten_next = False
    _restarts = 0

    async def request(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await super().request(*args, **kwargs)
        except ModelHTTPError as exc:
            if (
                not _session_lost(exc)
                or self.restart_session is None
                or self._restarts >= _MAX_SESSION_RESTARTS
            ):
                raise
            self._restarts += 1
            old, new = self.restart_session()
            logger.warning("Gateway lost session %s; continuing in %s", old, new)
            observer = getattr(self, "wire_observer", None)
            if observer is not None:
                observer.event(
                    "gateway_session_restarted",
                    lost_session=old,
                    session=new,
                    restart=self._restarts,
                )
        self._flatten_next = True
        try:
            return await super().request(*args, **kwargs)
        finally:
            self._flatten_next = False

    async def _map_messages(
        self,
        messages: Sequence[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        *,
        model_settings: ModelSettings | None = None,
    ) -> list[ChatCompletionMessageParam]:
        mapped = await super()._map_messages(
            messages, model_request_parameters, model_settings=model_settings,
        )
        if self._flatten_next:
            return _flatten_history(mapped)
        last_response = next(
            (i for i in range(len(mapped) - 1, -1, -1)
             if mapped[i]["role"] == "assistant"),
            None,
        )
        if last_response is None:
            return mapped
        instructions = [
            m for m in mapped[:last_response]
            if m["role"] in {"system", "developer"}
        ]
        # Keep the outstanding call so the gateway validates its ID, name and
        # arguments; preserve every following tool result and image attachment.
        start = last_response if mapped[last_response].get("tool_calls") else last_response + 1
        return instructions + mapped[start:]


class CodexGatewayRuntime:
    """Keep the gateway's suspended tool turn local to one solve invocation."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        max_tokens: int,
        timeout_s: int | None = None,
        dashboard: Any = None,
        recorder: Any = None,
        no_images: bool = False,
        image_archive: Any = None,
    ):
        if max_tokens != 0:
            raise ValueError(
                "CodexGateway cannot enforce an output token cap; "
                "pass --max-tokens 0 explicitly."
            )
        if not model:
            raise ValueError("codexgateway: requires an actual model id")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not self._base_url:
            raise ValueError("CodexGateway requires --base-url or OPENAI_BASE_URL")
        self._api_key = os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("CodexGateway requires OPENAI_API_KEY")
        self._model = model
        self._timeout_s = timeout_s
        self._image_archive = image_archive
        self._options = {
            "max_tokens": 0,
            "dashboard": dashboard,
            "recorder": recorder,
            "no_images": no_images,
            "image_archive": image_archive,
        }

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int | None,
        input_queue: queue.Queue[str | None] | None = None,
    ) -> PlannerResult:
        return asyncio.run(
            self._solve(
                system_prompt=system_prompt,
                user_message=user_message,
                toolkit=toolkit,
                max_turns=max_turns,
                input_queue=input_queue,
            )
        )

    async def _solve(self, **kwargs: Any) -> PlannerResult:
        # A runtime can be reused by nested planning calls on different event
        # loops. Both the HTTP pool and the gateway session belong to this call.
        session_id = "embrain-" + uuid4().hex
        base_url = self._base_url.rstrip("/") + "/"
        headers = {
            "Authorization": "Bearer " + self._api_key,
            "X-Codex-Session-ID": session_id,
        }
        async with httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=330,
            trust_env=False,
            event_hooks=(
                {"request": [self._image_archive.on_request],
                 "response": []}
                if self._image_archive is not None else None
            ),
        ) as http:
            observer = RequestObserver(self._image_archive, recorder=self._options["recorder"])
            observer.install(http)
            client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=base_url,
                http_client=http,
                max_retries=0,
            )
            model = _CodexGatewayChatModel(
                self._model, provider=OpenAIProvider(openai_client=client)
            )
            model.wire_observer = observer
            model.wire_telemetry = True
            model.handles_request_budget = True
            sessions = [session_id]

            def restart_session() -> tuple[str, str]:
                sessions.append("embrain-" + uuid4().hex)
                http.headers["X-Codex-Session-ID"] = sessions[-1]
                return sessions[-2], sessions[-1]

            model.restart_session = restart_session
            runtime = ApiAgentRuntime(model=model, **self._options)
            try:
                return await asyncio.wait_for(
                    runtime._solve(**kwargs),
                    timeout=self._timeout_s,
                )
            finally:
                observer.summarize()
                # finish and tool budgets can stop locally while the gateway is
                # still waiting for a tool result. Release that pending turn too.
                for session in sessions:
                    try:
                        response = await http.delete(
                            "codex/sessions/" + session,
                            timeout=10,
                        )
                        if response.status_code != 404:
                            response.raise_for_status()
                    except httpx.HTTPError as exc:
                        logger.warning(
                            "Could not release gateway session %s (%s)",
                            session,
                            type(exc).__name__,
                        )
