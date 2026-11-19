"""Observe actual wire responses and bounded retries without guessing usage.

HTTP attempts, including failures, own the episode request charge. Missing
provider fields remain null; selected request models never fill response models.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx

from rpent.episode.liveness import current_budget
from rpent.recording.media import atomic_write_bytes

TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
)
HEADER_FIELDS = (
    "x-request-id",
    "request-id",
    "x-model",
    "x-served-model",
    "openai-model",
    "openai-processing-ms",
)


def _counter(value):
    return value if type(value) is int and value >= 0 else None


def response_metadata(body: dict, headers: Any) -> dict:
    """Normalize Chat/Responses usage, retaining absence separately from zero."""
    usage = body.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    input_details = (
        usage.get("input_tokens_details", usage.get("prompt_tokens_details")) or {}
    )
    output_details = (
        usage.get("output_tokens_details", usage.get("completion_tokens_details")) or {}
    )
    tokens = {
        "input_tokens": _counter(usage.get("input_tokens", usage.get("prompt_tokens"))),
        "cached_input_tokens": _counter(input_details.get("cached_tokens")),
        "output_tokens": _counter(
            usage.get("output_tokens", usage.get("completion_tokens"))
        ),
        "reasoning_tokens": _counter(output_details.get("reasoning_tokens")),
    }
    model = body.get("model")
    model = model if isinstance(model, str) and model else None
    missing = {k: "gateway_not_provided" for k, v in tokens.items() if v is None}
    if model is None:
        missing["model"] = "gateway_not_provided"
    return {
        "model": model,
        "model_source": "response_body" if model else None,
        "usage": tokens,
        "missing_fields": missing,
        "response_headers": {k: headers[k] for k in HEADER_FIELDS if k in headers},
    }


def summarize_responses(rows: list[dict]) -> dict:
    """Sum known values, withholding a total if any response lacks that field."""
    tokens = {}
    for key in TOKEN_FIELDS:
        values = [r.get("usage", {}).get(key) for r in rows]
        known = [v for v in values if _counter(v) is not None]
        missing = len(values) - len(known)
        tokens[key] = {
            "known_total": sum(known),
            "missing_responses": missing,
            "total": None if missing else sum(known),
        }
    latencies = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    return {
        "schema_version": 1,
        "observable_api_requests": len(rows),
        "models": dict(Counter(r.get("model") or "not_provided" for r in rows)),
        "tokens": tokens,
        "average_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "scope": "observable HTTP responses, including failed attempts; provider-native hidden requests are opaque",
    }


class RequestObserver:
    """Install around HTTP send; SDK retries must remain disabled.

    Three attempts total (initial + at most two retries), with 1/2 second
    exponential backoff. No 4xx, connection errors or cancellation retry.
    Stateful gateways may invalidate their session on a 504: a subsequent 4xx
    is retained and stops immediately; never invent/replay physical tool output.
    """

    def __init__(self, archive=None, *, recorder=None, sleep=asyncio.sleep):
        self.archive = archive
        self.recorder = recorder or getattr(archive, "recorder", None)
        self.sleep = sleep
        self.rows = []

    def install(self, http):
        self._send = http.send
        http.send = self.send

    def event(self, kind, **payload):
        if self.recorder is not None:
            self.recorder.record(kind, **payload)

    async def send(self, request, **kwargs):
        if request.method != "POST":
            return await self._send(request, **kwargs)
        logical_id = uuid4().hex
        try:
            requested_model = json.loads(await request.aread()).get("model")
        except (ValueError, AttributeError):
            requested_model = None
        request.extensions["rpent_requested_model"] = requested_model
        for attempt in range(1, 4):
            budget = current_budget()
            if budget is not None:
                budget.reserve_request()
            # HTTP hooks create an independent immutable archive per attempt.
            request.extensions.pop("rpent_llm_request_id", None)
            started = time.monotonic()
            self.event(
                "api_request",
                logical_request_id=logical_id,
                attempt=attempt,
                request_scope="observable_http",
                budget=budget.snapshot() if budget else None,
            )
            response, failure = None, None
            try:
                response = await self._send(request, **kwargs)
                # Standard episode requests are nonstreaming. An explicitly
                # streaming response is observed only after EOF by this wrapper.
                await response.aread()
            except httpx.TimeoutException as exc:
                failure = exc
            except asyncio.CancelledError:
                self.observe(request, None, started, logical_id, attempt, "cancelled")
                raise
            except httpx.HTTPError as exc:
                self.observe(
                    request, None, started, logical_id, attempt, type(exc).__name__
                )
                raise
            status = response.status_code if response is not None else None
            self.observe(
                request,
                response,
                started,
                logical_id,
                attempt,
                type(failure).__name__ if failure else None,
            )
            retryable = failure is not None or (
                status is not None and 500 <= status <= 599
            )
            if not retryable or attempt == 3:
                if failure is not None:
                    raise failure
                return response
            delay = 2 ** (attempt - 1)
            if response is not None:
                await response.aclose()
            self.event(
                "api_retry",
                logical_request_id=logical_id,
                request_id=request.extensions.get("rpent_llm_request_id"),
                status_code=status,
                error=type(failure).__name__ if failure else None,
                failed_attempt=attempt,
                retry_attempt=attempt,
                next_attempt=attempt + 1,
                backoff_s=delay,
            )
            waited = time.monotonic()
            await self.sleep(delay)
            self.event(
                "api_retry_wait_finished",
                logical_request_id=logical_id,
                retry_attempt=attempt,
                backoff_s=delay,
                waited_s=time.monotonic() - waited,
            )
        raise AssertionError("bounded retry loop must return or raise")

    def observe(self, request, response, started, logical_id, attempt, error=None):
        body = {}
        if response is not None:
            try:
                body = response.json()
            except (ValueError, UnicodeError):
                # SSE may carry model/usage in its final event. Retain only
                # telemetry fields, never response reasoning/text in this log.
                for line in response.text.splitlines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        part = json.loads(line[6:])
                    except ValueError:
                        continue
                    if isinstance(part, dict):
                        part = part.get("response", part)
                        body.update(
                            {k: v for k, v in part.items() if k in {"model", "usage"}}
                        )
        if not isinstance(body, dict):
            body = {}
        row = {
            "schema_version": 2,
            "request_id": request.extensions.get("rpent_llm_request_id") or uuid4().hex,
            "logical_request_id": logical_id,
            "attempt": attempt,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "status": "http_response_received"
            if response is not None
            else "http_request_failed",
            "status_code": response.status_code if response is not None else None,
            "requested_model": request.extensions.get("rpent_requested_model"),
            "error": error,
            "latency_s": time.monotonic() - started,
            **response_metadata(body, response.headers if response is not None else {}),
        }
        self.rows.append(row)
        if self.archive is not None:
            self.archive._append("responses.jsonl", row)
        self.event("api_response", **row)
        self.event(
            "usage_updated",
            **row["usage"],
            model=row["model"],
            model_source=row["model_source"],
            latency_s=row["latency_s"],
            request_id=row["request_id"],
            logical_request_id=logical_id,
            attempt=attempt,
            missing_fields=row["missing_fields"],
            usage_scope="response",
        )

    def summarize(self):
        """Refresh episode journal summary at solve exit, including nested solves."""
        if self.archive is None:
            return summarize_responses(self.rows)
        path = self.archive.root / "llm_inputs/responses.jsonl"
        rows = (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )
        result = summarize_responses(rows)
        atomic_write_bytes(
            self.archive.root / "llm_inputs/usage-summary.json",
            (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode(),
        )
        self.event("usage_summary", **result)
        return result
