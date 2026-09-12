"""Unit tests for loadtest/bench.py (pure logic; no server, no GPU)."""

from __future__ import annotations


def test_make_payload_baseline_shape(bench):
    p = bench.make_payload("baseline", 64)
    assert p["prompt"] == bench.PROMPT
    assert p["max_tokens"] == 64
    assert p["temperature"] == bench.TEMPERATURE
    assert p["top_p"] == bench.TOP_P
    assert "model" not in p  # baseline has no model field
    assert "messages" not in p


def test_make_payload_vllm_shape(bench):
    p = bench.make_payload("vllm", 128, model="Qwen/Qwen2.5-3B-Instruct-AWQ")
    assert p["model"] == "Qwen/Qwen2.5-3B-Instruct-AWQ"
    assert p["max_tokens"] == 128
    assert p["messages"] == [{"role": "user", "content": bench.PROMPT}]


def test_make_payload_default_model_is_canonical(bench):
    p = bench.make_payload("vllm", 32)
    assert p["model"] == bench.BASELINE_MODEL == "Qwen/Qwen2.5-0.5B-Instruct"


def test_extract_tokens_baseline(bench):
    resp = {"tokens_generated": 42}
    assert bench.extract_tokens("baseline", {}, resp, 200) == 42


def test_extract_tokens_vllm_usage(bench):
    resp = {"usage": {"completion_tokens": 17}}
    assert bench.extract_tokens("vllm", {}, resp, 200) == 17


def test_extract_tokens_non_200_is_zero(bench):
    assert bench.extract_tokens("vllm", {}, {"usage": {"completion_tokens": 9}}, 500) == 0
    assert bench.extract_tokens("baseline", {}, {"tokens_generated": 9}, 503) == 0


def test_extract_tokens_missing_usage_defaults_zero(bench):
    assert bench.extract_tokens("vllm", {}, {}, 200) == 0


def test_pretty_contains_key_metrics(bench):
    row = {
        "mode": "vllm", "concurrency": 4, "requests": 8, "stream": False,
        "out_tok_s": 123.4, "lat_p50_ms": 10.0, "lat_p95_ms": 20.0,
        "lat_p99_ms": 30.0, "errors": 0, "total_out_tokens": 512,
    }
    line = bench.pretty(row)
    assert "out_tok/s" in line and "p50=" in line and "errs=0" in line


def test_pretty_stream_includes_ttft_itl(bench):
    row = {
        "mode": "vllm", "concurrency": 2, "requests": 4, "stream": True,
        "out_tok_s": 99.0, "lat_p50_ms": 10.0, "lat_p95_ms": 20.0,
        "lat_p99_ms": 30.0, "errors": 0, "total_out_tokens": 256,
        "ttft_p50_ms": 25.0, "ttft_p95_ms": 40.0, "itl_p50_ms": 8.0,
    }
    line = bench.pretty(row)
    assert "TTFT" in line and "ITL" in line


def test_cli_rejects_stream_for_baseline(bench):
    import asyncio

    # --stream is only valid with --mode vllm; argparse should exit(2).
    import pytest

    with pytest.raises(SystemExit):
        asyncio.run(
            bench.amain(
                ["--url", "http://x", "--mode", "baseline", "--stream"]
            )
        )


def test_cli_parses_defaults(bench, monkeypatch):
    import asyncio

    captured = {}

    async def fake_run_sweep(mode, url, concurrency, n_requests, max_tokens=128, model=None, stream=False):
        captured.update(
            mode=mode, url=url, concurrency=concurrency,
            requests=n_requests, max_tokens=max_tokens, model=model, stream=stream,
        )
        return {"mode": mode, "concurrency": concurrency, "requests": n_requests,
                "stream": stream, "out_tok_s": 0.0, "lat_p50_ms": 0.0,
                "lat_p95_ms": 0.0, "lat_p99_ms": 0.0, "errors": 0,
                "total_out_tokens": 0}

    monkeypatch.setattr(bench, "run_sweep", fake_run_sweep)
    rc = asyncio.run(bench.amain(["--url", "http://x/v1/chat/completions", "--concurrency", "3", "--requests", "6"]))
    assert rc == 0
    assert captured["mode"] == "vllm"
    assert captured["concurrency"] == 3
    assert captured["requests"] == 6
    assert captured["model"] == bench.BASELINE_MODEL
    assert captured["stream"] is False
