"""Wire telemetry distinguishes absent fields and charges every retry."""

import asyncio
import json

import httpx
import pytest

from rpent.episode.liveness import EpisodeBudget, EpisodeBudgetExceeded, budget_scope
from rpent.recording.llm_images import LLMImageArchive
from rpent.runtime.telemetry import (
    RequestObserver,
    response_metadata,
    summarize_responses,
)


class Recorder:
    def __init__(self):
        self.events = []

    def record(self, event, **payload):
        self.events.append((event, payload))


def test_response_metadata_does_not_invent_missing_usage():
    m = response_metadata({"model": "actual"}, {"x-request-id": "r"})
    assert m["model"] == "actual"
    assert all(value is None for value in m["usage"].values())
    assert m["missing_fields"]["input_tokens"] == "gateway_not_provided"
    assert m["response_headers"] == {"x-request-id": "r"}
    assert response_metadata({}, {})["model"] is None


@pytest.mark.parametrize("responses", [False, True])
def test_all_four_token_fields_are_parsed(responses):
    usage = (
        {
            "input_tokens": 101,
            "output_tokens": 31,
            "input_tokens_details": {"cached_tokens": 17},
            "output_tokens_details": {"reasoning_tokens": 11},
        }
        if responses
        else {
            "prompt_tokens": 101,
            "completion_tokens": 31,
            "prompt_tokens_details": {"cached_tokens": 17},
            "completion_tokens_details": {"reasoning_tokens": 11},
        }
    )
    m = response_metadata({"model": "served", "usage": usage}, {})
    assert m["usage"] == {"input_tokens":101, "output_tokens":31, "cached_input_tokens":17, "reasoning_tokens":11}
    assert not m["missing_fields"]


@pytest.mark.parametrize(
    "codes,count", [([504, 200], 2), ([504, 504, 504], 3), ([400], 1)]
)
def test_bounded_retries_archive_and_budget(tmp_path, codes, count):
    sent, waits, recorder = [], [], Recorder()
    archive = LLMImageArchive(tmp_path, recorder=recorder)
    budget = EpisodeBudget(max_requests=500, max_seconds=60)

    async def sleep(seconds):
        waits.append(seconds)

    def handle(request):
        sent.append(request.content)
        return httpx.Response(
            codes[len(sent) - 1],
            json={
                "model": "served",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle),
            event_hooks={"request": [archive.on_request]},
        ) as http:
            observer = RequestObserver(archive, recorder=recorder, sleep=sleep)
            observer.install(http)
            with budget_scope(budget):
                response = await http.post(
                    "http://test/v1/chat/completions",
                    json={
                        "model": "requested",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
            observer.summarize()
            return response

    assert asyncio.run(run()).status_code == codes[-1]
    assert len(sent) == count
    assert len(set(sent)) == 1
    assert waits == [1, 2][: count - 1]
    assert budget.snapshot()["llm_requests"] == count
    rows = [
        json.loads(line)
        for line in (tmp_path / "llm_inputs/responses.jsonl").read_text().splitlines()
    ]
    assert len(rows) == count
    assert len({row["request_id"] for row in rows}) == count
    assert all(row["model"] == "served" and row["latency_s"] >= 0 for row in rows)
    retries = [p for e, p in recorder.events if e == "api_retry"]
    assert [r["retry_attempt"] for r in retries] == list(range(1, count))
    assert all(r["status_code"] == 504 for r in retries)
    assert any(
        e == "usage_updated" and p["input_tokens"] == 10 for e, p in recorder.events
    )


def test_timeout_retries_stop_at_episode_cap(tmp_path):
    calls = []

    async def fail(request):
        calls.append(1)
        raise httpx.ReadTimeout("timeout", request=request)

    async def sleep(seconds):
        pass

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as http:
            observer = RequestObserver(LLMImageArchive(tmp_path), sleep=sleep)
            observer.install(http)
            budget = EpisodeBudget(max_requests=2, max_seconds=60)
            with budget_scope(budget), pytest.raises(EpisodeBudgetExceeded):
                await http.post("http://test/v1/chat/completions", json={"model": "x"})
            assert budget.snapshot()["llm_requests"] == 2

    asyncio.run(run())
    assert len(calls) == 2


def test_partial_totals_are_explicit():
    summary = summarize_responses(
        [
            {"model": "a", "latency_s": 2, "usage": {"input_tokens": None}},
            {"model": "b", "latency_s": 4, "usage": {"input_tokens": 9}},
        ]
    )
    assert summary["tokens"]["input_tokens"] == {
        "known_total": 9,
        "missing_responses": 1,
        "total": None,
    }
    assert summary["models"] == {"a": 1, "b": 1}
    assert summary["average_latency_s"] == 3
