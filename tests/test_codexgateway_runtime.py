"""Exercise the API wire format needed by a Codex SDK gateway."""

import asyncio
import base64
import json

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from rpent.cli.main import _build_argparser
from rpent.envs import get_env_spec
from rpent.runtime.api import _build_model_settings


@pytest.mark.parametrize("limit", [0, 8192])
def test_chat_wire_preserves_images_and_tool_results_with_explicit_token_limit(limit):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        message = {"role": "assistant", "content": "done"}
        finish = "stop"
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "gateway-test-call",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"value":7}'},
                    }
                ],
            }
            finish = "tool_calls"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-5.6-luna",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    def echo(value: int) -> str:
        return str(value)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            client = AsyncOpenAI(
                api_key="test-only",
                base_url="http://gateway.invalid/v1",
                http_client=http,
            )
            model = OpenAIChatModel(
                "gpt-5.6-luna",
                provider=OpenAIProvider(openai_client=client),
            )
            agent = Agent(
                model,
                instructions="Use echo.",
                tools=[echo],
                model_settings=_build_model_settings(model, limit),
                capabilities=[Thinking(effort="high")],
            )
            return await agent.run(
                [
                    "Read the image and echo 7.",
                    BinaryContent(
                        base64.b64decode(
                            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
                            "/x8AAwMCAO+aZAAAAABJRU5ErkJggg=="
                        ),
                        media_type="image/png",
                    ),
                ]
            )

    result = asyncio.run(run())
    assert result.output == "done"
    assert len(requests) == 2
    for body in requests:
        assert body["model"] == "gpt-5.6-luna"
        assert body["reasoning_effort"] == "high"
        assert body["tools"][0]["function"]["name"] == "echo"
        if limit:
            assert body["max_completion_tokens"] == limit
        else:
            assert "max_completion_tokens" not in body
            assert "max_tokens" not in body
    assert requests[0]["messages"][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "gateway-test-call",
        "content": "7",
    }


def test_robocasa_launch_preserves_explicit_zero_api_token_limit():
    preset = get_env_spec("robocasa").launch_preset()
    values = preset.normalize(
        "hybrid",
        {
            "task": "CloseDrawer",
            "planner": "api",
            "vla": "pi05",
            "model": "openai-chat:gpt-5.6-luna",
            "max_tokens": 0,
        },
    )
    assert values["max_tokens"] == 0
    assert values["planning_profile"] == "robocasa-horizon-gated-v1"


def test_negative_api_token_limit_is_rejected_before_launch():
    with pytest.raises(SystemExit):
        _build_argparser().parse_args(["--max-tokens", "-1"])


def test_gateway_factory_isolates_and_cleans_each_solve(monkeypatch, tmp_path):
    from rpent.runtime.api import ApiAgentRuntime
    from rpent.runtime.base import PlannerResult
    from rpent.runtime.factory import build_runtime

    requests = []
    original_client = httpx.AsyncClient

    def handle(request):
        requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(200, json={"deleted": True})
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-5.6-luna",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    class MockClient(original_client):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gateway.invalid/v1")

    async def solve(self, **kwargs):
        result = await Agent(
            self._model,
            model_settings=_build_model_settings(
                self._model,
                self._max_tokens,
            ),
        ).run("hello")
        return PlannerResult(messages=[{"text": result.output}])

    monkeypatch.setattr(ApiAgentRuntime, "_solve", solve)
    runtime = build_runtime(
        "api",
        output_dir=tmp_path,
        recipe_tag="episode",
        env_name="robocasa",
        model="codexgateway:gpt-5.6-luna",
        max_tokens=0,
    )
    for _ in range(2):
        assert runtime.solve(
            system_prompt="", user_message="hello", toolkit=None, max_turns=1
        ).messages == [{"text": "done"}]
    assert [r.method for r in requests] == ["POST", "DELETE", "POST", "DELETE"]
    ids = [r.headers["X-Codex-Session-ID"] for r in requests]
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]
    for post, delete in zip(requests[::2], requests[1::2], strict=True):
        body = json.loads(post.content)
        assert body["model"] == "gpt-5.6-luna"
        assert "max_completion_tokens" not in body
        assert post.url.path == "/v1/chat/completions"
        assert (
            delete.url.path
            == "/v1/codex/sessions/" + post.headers["X-Codex-Session-ID"]
        )
        assert delete.headers["Authorization"] == "Bearer test-only"


def test_gateway_requires_explicit_unsupported_token_cap_opt_out(monkeypatch, tmp_path):
    from rpent.runtime.factory import build_runtime

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    with pytest.raises(ValueError, match="--max-tokens 0"):
        build_runtime(
            "api",
            output_dir=tmp_path,
            recipe_tag="episode",
            env_name="robocasa",
            base_url="http://gateway.invalid/v1",
            model="codexgateway:gpt-5.6-luna",
        )


@pytest.mark.parametrize(
    "failure", [RuntimeError("test error"), asyncio.CancelledError()]
)
def test_gateway_cleans_session_on_failure(monkeypatch, tmp_path, failure):
    from rpent.runtime.api import ApiAgentRuntime
    from rpent.runtime.factory import build_runtime

    deleted = []
    original_client = httpx.AsyncClient

    def handle(request):
        deleted.append(request)
        return httpx.Response(404, json={"error": {"message": "not found"}})

    class MockClient(original_client):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")

    async def solve(self, **kwargs):
        raise failure

    monkeypatch.setattr(ApiAgentRuntime, "_solve", solve)
    runtime = build_runtime(
        "api",
        output_dir=tmp_path,
        recipe_tag="episode",
        env_name="robocasa",
        base_url="http://gateway.invalid/v1",
        model="codexgateway:gpt-5.6-luna",
        max_tokens=0,
    )
    with pytest.raises(type(failure)):
        runtime.solve(system_prompt="", user_message="hello", toolkit=None, max_turns=1)
    assert len(deleted) == 1 and deleted[0].method == "DELETE"


def test_dashboard_preserves_zero_token_limit_in_cli_conversion():
    from rpent.ui.app import _config_argv

    assert _config_argv(_build_argparser(), {"max_tokens": 0}) == ["--max-tokens", "0"]


def test_argparse_schema_declares_nonnegative_token_limit():
    from rpent.ui.routes.simulators import _parser_schema

    value = _parser_schema(_build_argparser())["properties"]["max_tokens"]
    assert value["type"] == "integer"
    assert value["minimum"] == 0


def test_gateway_timeout_cancels_and_cleans_on_python310(monkeypatch, tmp_path):
    from rpent.runtime.api import ApiAgentRuntime
    from rpent.runtime.factory import build_runtime

    requests = []
    cancelled = []
    original_client = httpx.AsyncClient

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"deleted": True})

    class MockClient(original_client):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    # Python 3.10 is supported by the project and has no asyncio.timeout.
    monkeypatch.delattr(asyncio, "timeout", raising=False)

    async def solve(self, **kwargs):
        try:
            await asyncio.sleep(3600)
        finally:
            cancelled.append(True)

    monkeypatch.setattr(ApiAgentRuntime, "_solve", solve)
    runtime = build_runtime(
        "api",
        output_dir=tmp_path,
        recipe_tag="episode",
        env_name="robocasa",
        model="codexgateway:gpt-5.6-luna",
        base_url="http://gateway.invalid/v1",
        max_tokens=0,
        planner_timeout_s=0.01,
    )
    with pytest.raises(asyncio.TimeoutError):
        runtime.solve(system_prompt="", user_message="hello", toolkit=None, max_turns=1)
    assert cancelled == [True]
    assert len(requests) == 1 and requests[0].method == "DELETE"


@pytest.mark.parametrize("legacy", [False, True])
def test_gateway_long_tool_session_sends_only_current_exchange(monkeypatch, tmp_path, legacy):
    import io

    from PIL import Image

    from rpent.recording.llm_images import LLMImageArchive
    from rpent.runtime.codexgateway import CodexGatewayRuntime
    from rpent.tools.toolkit import Toolkit

    requests = []
    payload = "sensor evidence " * 1000
    rounds = 80
    pixels = []
    for color in ["red", "green", "blue"]:
        stream = io.BytesIO()
        Image.new("RGB", (256, 256), color).save(stream, format="PNG")
        pixels.append(stream.getvalue())
    cameras = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    fields = ["_image_bytes", "_image_right_bytes", "_image_wrist_bytes"]
    original_client = httpx.AsyncClient

    def handle(request):
        if request.method == "DELETE":
            return httpx.Response(200, json={"deleted": True})
        body = json.loads(request.content)
        requests.append(body)
        if len(request.content) > 200_000:
            return httpx.Response(400, json={"error": {
                "message": "The prompt exceeds the configured size limit.",
                "type": "invalid_request_error", "code": "context_length_exceeded",
            }})
        n = len(requests)
        if n > 1:
            messages = body["messages"]
            tools = [m for m in messages if m["role"] == "tool"]
            assert len(tools) == 1
            assert tools[0]["tool_call_id"] == f"call-{n-1}"
            assert payload in tools[0]["content"]
            assert any(
                isinstance(m.get("content"), list)
                and any(p.get("type") == "image_url" for p in m["content"])
                for m in messages if m["role"] == "user"
            )
        return httpx.Response(200, json={
            "id": f"chatcmpl-{n}", "object": "chat.completion", "created": n,
            "model": "gpt-5.6-luna",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": f"call-{n}", "type": "function", "function": {
                    "name": "sample" if n <= rounds else "finish",
                    "arguments": "{}" if n <= rounds else json.dumps({
                        "status": "failure", "summary": "offline test completed",
                    }),
                }}],
            }}],
        })

    class MockClient(original_client):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    toolkit = Toolkit()
    toolkit.add_tool("sample", {
        "name": "sample", "description": "Collect evidence.",
        "input_schema": {"type": "object", "properties": {}},
    }, lambda: {"evidence": payload, "step": len(requests),
                "_image_camera_ids": dict(zip(cameras, fields)), **dict(zip(fields, pixels))})
    if legacy:
        original_execute = toolkit.execute_tool

        def unlabeled(*args, **kwargs):
            result = original_execute(*args, **kwargs)
            result.content_blocks = [p for p in result.content_blocks
                                     if not (p.get("type") == "text" and p.get("text", "").startswith("Image "))]
            return result

        monkeypatch.setattr(toolkit, "execute_tool", unlabeled)
    runtime = CodexGatewayRuntime(
        model="gpt-5.6-luna", base_url="http://gateway.invalid/v1", max_tokens=0,
        image_archive=LLMImageArchive(tmp_path),
    )
    result = runtime.solve(
        system_prompt="Preserve the original task and all evidence.",
        user_message="offline task", toolkit=toolkit, max_turns=None,
    )
    assert result.error is None
    assert len(requests) == rounds + 1
    assert len([m for m in result.messages if m["role"] == "tool"]) == rounds + 1
    assert all(len(json.dumps(r)) < 200_000 for r in requests)
    archived = [json.loads(line) for line in (tmp_path / "llm_inputs/requests.jsonl").read_text().splitlines()]
    assert len(archived) == len(requests)
    assert archived[0]["images"] == []
    for n, row in enumerate(archived[1:], 1):
        assert row["capture_stage"] == "http_request_body"
        # Previous exchanges remain in the gateway thread, not the next HTTP body.
        assert len(row["images"]) == 3
        assert [image["camera_id"] for image in row["images"]] == cameras
        assert all(image["observation_step"] == n for image in row["images"])
        assert [(tmp_path / image["relative_path"]).read_bytes() for image in row["images"]] == pixels
    assert len(list((tmp_path / "llm_inputs/images").iterdir())) == 3


def _completion(n, name, arguments="{}"):
    return {
        "id": f"chatcmpl-{n}", "object": "chat.completion", "created": n,
        "model": "gpt-5.6-luna",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": f"call-{n}", "type": "function",
                            "function": {"name": name, "arguments": arguments}}],
        }}],
    }


def _gateway(monkeypatch, handle):
    original_client = httpx.AsyncClient

    class MockClient(original_client):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")


@pytest.mark.parametrize("refusal", [
    "Tool history cannot be replayed into a new session.",  # session exists without its thread
    "No pending tool call in this session.",  # session id unknown to the gateway (1.2.0 D10, 2026-10-12)
])
def test_gateway_lost_session_continues_with_flattened_history(monkeypatch, refusal):
    from rpent.runtime.codexgateway import CodexGatewayRuntime
    from rpent.tools.toolkit import Toolkit

    posts, deletes = [], []
    lost = {"error": {"message": refusal,
                      "type": "invalid_request_error", "code": "tool_call_mismatch"}}
    finish = json.dumps({"status": "failure", "summary": "offline test completed"})

    def handle(request):
        if request.method == "DELETE":
            deletes.append(request.url.path.rsplit("/", 1)[-1])
            return httpx.Response(404 if len(deletes) == 1 else 200, json={})
        posts.append((request.headers["X-Codex-Session-ID"], json.loads(request.content)))
        n = len(posts)
        if n == 1:
            return httpx.Response(200, json=_completion(1, "sample"))
        if n == 2:  # the upstream turn fails; the gateway discards the session
            return httpx.Response(502, json={"error": {"message": "upstream failed"}})
        if n == 3:  # the retried tool exchange reaches a new, empty session
            return httpx.Response(400, json=lost)
        return httpx.Response(200, json=_completion(n, "finish", finish))

    _gateway(monkeypatch, handle)
    toolkit = Toolkit()
    toolkit.add_tool("sample", {
        "name": "sample", "description": "Collect evidence.",
        "input_schema": {"type": "object", "properties": {}},
    }, lambda: {"evidence": "bowl held at 0.12 m"})
    runtime = CodexGatewayRuntime(
        model="gpt-5.6-luna", base_url="http://gateway.invalid/v1", max_tokens=0,
    )
    result = runtime.solve(system_prompt="Keep the bowl.", user_message="offline task",
                           toolkit=toolkit, max_turns=None)
    assert result.error is None
    assert len(posts) == 4
    first, restarted = posts[0][0], posts[3][0]
    assert posts[1][0] == posts[2][0] == first != restarted
    messages = posts[3][1]["messages"]
    assert not any(m["role"] == "tool" or m.get("tool_calls") for m in messages)
    assert messages[-1]["role"] == "user"
    assert "[called sample({})]" in messages[-2]["content"]
    assert "[result of call-1]" in messages[-1]["content"]
    assert "bowl held at 0.12 m" in messages[-1]["content"]
    assert deletes == [first, restarted]


@pytest.mark.parametrize("message,code", [
    ("Tool result ID does not match the pending call.", "tool_call_mismatch"),
    ("This tool result was already consumed.", "tool_call_mismatch"),
    ("No pending tool call in this session.", "some_other_code"),
])
def test_gateway_other_tool_mismatch_is_not_masked(monkeypatch, message, code):
    from rpent.runtime.codexgateway import CodexGatewayRuntime
    from rpent.tools.toolkit import Toolkit

    posts = []

    def handle(request):
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        posts.append(request)
        if len(posts) == 1:
            return httpx.Response(200, json=_completion(1, "sample"))
        return httpx.Response(400, json={"error": {
            "message": message, "type": "invalid_request_error", "code": code}})

    _gateway(monkeypatch, handle)
    toolkit = Toolkit()
    toolkit.add_tool("sample", {
        "name": "sample", "description": "Collect evidence.",
        "input_schema": {"type": "object", "properties": {}},
    }, lambda: {"evidence": "x"})
    runtime = CodexGatewayRuntime(
        model="gpt-5.6-luna", base_url="http://gateway.invalid/v1", max_tokens=0,
    )
    result = runtime.solve(system_prompt="", user_message="offline task",
                           toolkit=toolkit, max_turns=None)
    assert result.error is not None and message.rstrip(".") in str(result.error)
    assert len(posts) == 2


def test_restarted_session_history_fits_the_gateway_prompt_limit():
    from rpent.runtime.codexgateway import _flatten_history, _size

    big = "state " * 2_500  # about 15k characters per tool result, as in a long episode
    mapped = [{"role": "system", "content": "rules"}, {"role": "user", "content": "task"}]
    for n in range(120):
        mapped.append({"role": "assistant", "content": None, "tool_calls": [{
            "id": f"call-{n}", "type": "function", "function": {"name": "sample", "arguments": "{}"}}]})
        mapped.append({"role": "tool", "tool_call_id": f"call-{n}", "content": big + str(n)})
    flat = _flatten_history(mapped)
    assert sum(_size(m) for m in flat) <= 160_000
    assert flat[:2] == mapped[:2]
    assert "earlier messages omitted" in flat[2]["content"]
    assert flat[-1]["role"] == "user" and flat[-1]["content"].endswith("119")
    assert not any(m["role"] == "tool" or m.get("tool_calls") for m in flat)
