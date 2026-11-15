"""Offline tests: data math on synthetic prices, and the Claude request shape
against a mocked HTTP transport (no API key or network needed)."""
import json

import httpx2 as httpx  # the HTTP library anthropic>=1.0 uses
import anthropic

from stock_analyzer.analyzer import analyze
from stock_analyzer.sample import sample_snapshot


def test_snapshot_has_core_metrics():
    s = sample_snapshot()
    for k in ("return_1y", "sma_200", "high_52w"):
        assert s.price[k] is not None
    for k in ("annualized_volatility", "max_drawdown", "beta_vs_benchmark_computed"):
        assert s.risk[k] is not None
    assert s.valuation["trailing_pe"] == 31.2
    assert 0.9 < s.risk["beta_vs_benchmark_computed"] < 1.6
    json.dumps(s.to_dict())  # must be serializable


def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def test_analyze_request_and_stream():
    captured = {}

    def handler(request: httpx.Request):
        captured["body"] = json.loads(request.content)
        captured["beta"] = request.headers.get("anthropic-beta")
        msg = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
               "content": [], "stop_reason": None, "stop_sequence": None,
               "usage": {"input_tokens": 10, "output_tokens": 0}}
        events = [
            {"type": "message_start", "message": msg},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "# DEMO report"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
             "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
        return httpx.Response(200, text=_sse(events), headers={"content-type": "text/event-stream"})

    client = anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    chunks = []
    report = analyze(sample_snapshot(), client=client, on_text=chunks.append)

    assert report == "# DEMO report"
    assert chunks == ["# DEMO report"]
    body = captured["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "medium"}
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in captured["beta"]
    assert "<snapshot>" in body["messages"][0]["content"]
