#!/usr/bin/env python3
"""
bench.py — Async concurrency throughput/latency client (Phases 2 & 3).

Measures aggregate output tokens/sec and latency percentiles by firing
`concurrency` identical generation requests at a FastAPI/OpenAI-compatible
endpoint. Supports BOTH targets so results are directly comparable:

  * baseline  (main-torch.py :8001)  -> POST /generate   (its Request/Response schema)
  * vLLM      (:8000)                -> POST /v1/chat/completions (OpenAI schema)

Phase 3 adds `--stream` (vLLM/OpenAI only): consume Server-Sent Events and
measure the canonical serving metrics TTFT (time-to-first-token) and ITL
(inter-token latency), in addition to aggregate tok/s and end-to-end latency.

Usage examples
--------------
# Baseline (single-request / static-KV reference):
python3 loadtest/bench.py --url http://127.0.0.1:8001/generate --mode baseline --concurrency 1 --requests 8

# vLLM concurrency sweep (non-streaming):
python3 loadtest/bench.py --url http://127.0.0.1:8000/v1/chat/completions --mode vllm --concurrency 8 --requests 15

# vLLM streaming: TTFT + ITL + tok/s at concurrency N:
python3 loadtest/bench.py --url http://127.0.0.1:8000/v1/chat/completions --mode vllm --stream --concurrency 4 --requests 12

Both modes use the SAME prompt / max_tokens / temperature for fairness.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import aiohttp

# --- Shared, identical workload across both targets ---------------------------
PROMPT = "Explain what an inference-serving engine does in two or three short sentences."
TEMPERATURE = 1.0
TOP_P = 0.9

BASELINE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def make_payload(mode: str, max_tokens: int, model: str = BASELINE_MODEL) -> dict:
    if mode == "baseline":
        return {
            "prompt": PROMPT,
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }
    # vllm / OpenAI chat completions
    return {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }


def extract_tokens(mode: str, payload: dict, resp: dict, status: int) -> int:
    """Return number of GENERATED (output) tokens for a successful response."""
    if status != 200:
        return 0
    if mode == "baseline":
        # main-torch returns tokens_generated (output tokens only)
        return int(resp.get("tokens_generated", 0))
    # vLLM: usage.completion_tokens is the generated-token count
    return int((resp.get("usage") or {}).get("completion_tokens", 0))


async def one_request(mode: str, session: aiohttp.ClientSession, url: str, payload: dict):
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload) as r:
            body = await r.json()
            status = r.status
    except Exception as exc:  # noqa: BLE001
        return {"lat": None, "tokens": 0, "ok": False, "err": repr(exc)}
    lat = (time.perf_counter() - t0) * 1000.0  # ms
    tokens = extract_tokens(mode, payload, body, status)
    return {"lat": lat, "tokens": tokens, "ok": status == 200, "err": None}


async def one_request_stream(session: aiohttp.ClientSession, url: str, payload: dict):
    """POST with stream=true (OpenAI SSE) and measure TTFT + ITL.

    TTFT = time from send to the first content-bearing chunk.
    ITL  = gaps (ms) between successive content-bearing chunks.
    Output token count = number of content chunks produced (1 token/chunk for
    vLLM's streaming API; `usage` is not reliably emitted mid-stream).
    """
    t0 = time.perf_counter()
    ttft_ms = None
    chunk_times: list[float] = []
    tokens = 0
    try:
        async with session.post(url, json=payload) as r:
            if r.status != 200:
                return {"lat": None, "tokens": 0, "ok": False, "err": f"HTTP {r.status}",
                        "ttft": None, "itl": None}
            async for raw in r.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    now = time.perf_counter()
                    if ttft_ms is None:
                        ttft_ms = (now - t0) * 1000.0
                    else:
                        chunk_times.append(now)
                    tokens += 1
    except Exception as exc:  # noqa: BLE001
        return {"lat": None, "tokens": 0, "ok": False, "err": repr(exc),
                "ttft": None, "itl": None}
    lat = (time.perf_counter() - t0) * 1000.0
    itls = [(chunk_times[i] - chunk_times[i - 1]) * 1000.0 for i in range(1, len(chunk_times))]
    itl_mean = (sum(itls) / len(itls)) if itls else None
    return {"lat": lat, "tokens": tokens, "ok": True, "err": None,
            "ttft": ttft_ms, "itl": itl_mean, "itls": itls}


async def run_sweep(mode: str, url: str, concurrency: int, n_requests: int,
                    max_tokens: int = 128, model: str = BASELINE_MODEL,
                    stream: bool = False) -> dict:
    payload = make_payload(mode, max_tokens, model)
    if stream:
        payload = {**payload, "stream": True}

    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(concurrency)

        async def worker():
            async with sem:
                if stream:
                    return await one_request_stream(session, url, payload)
                return await one_request(mode, session, url, payload)

        # Submit n_requests total, respecting concurrency cap.
        start = time.perf_counter()
        results = await asyncio.gather(*(worker() for _ in range(n_requests)))
        total_sec = time.perf_counter() - start

    ok = [r for r in results if r["ok"]]
    lats = sorted(r["lat"] for r in ok if r["lat"] is not None)
    total_out_tokens = sum(r["tokens"] for r in ok)
    errors = n_requests - len(ok)

    def pct(vals, p):
        if not vals:
            return None
        k = max(0, min(len(vals) - 1, int(round((p / 100.0) * (len(vals) - 1)))))
        return sorted(vals)[k]

    out = {
        "mode": mode,
        "url": url,
        "stream": stream,
        "concurrency": concurrency,
        "requests": n_requests,
        "errors": errors,
        "total_sec": total_sec,
        "total_out_tokens": total_out_tokens,
        "out_tok_s": (total_out_tokens / total_sec) if total_sec else 0.0,
        "lat_p50_ms": pct(lats, 50),
        "lat_p95_ms": pct(lats, 95),
        "lat_p99_ms": pct(lats, 99),
        "lat_mean_ms": (sum(lats) / len(lats)) if lats else None,
    }

    if stream:
        ttfts = [r["ttft"] for r in ok if r.get("ttft") is not None]
        itls = [r["itl"] for r in ok if r.get("itl") is not None]
        out.update({
            "ttft_p50_ms": pct(ttfts, 50),
            "ttft_p95_ms": pct(ttfts, 95),
            "ttft_mean_ms": (sum(ttfts) / len(ttfts)) if ttfts else None,
            "itl_p50_ms": pct(itls, 50),
            "itl_p95_ms": pct(itls, 95),
            "itl_mean_ms": (sum(itls) / len(itls)) if itls else None,
        })
    return out


def pretty(r: dict) -> str:
    head = (f"[{r['mode']:8s} conc={r['concurrency']:<2d} reqs={r['requests']:<3d}"
            f"{' stream' if r.get('stream') else ''}] ")
    body = (f"out_tok/s={r['out_tok_s']:8.1f}  "
            f"p50={r['lat_p50_ms']:.1f}ms p95={r['lat_p95_ms']:.1f}ms p99={r['lat_p99_ms']:.1f}ms ")
    if r.get("stream"):
        body += (f"TTFT p50={r['ttft_p50_ms']:.1f}ms p95={r['ttft_p95_ms']:.1f}ms  "
                 f"ITL p50={r['itl_p50_ms']:.1f}ms ")
    body += f"errs={r['errors']}  total_out_tokens={r['total_out_tokens']}"
    return head + body


async def amain(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="full endpoint URL")
    ap.add_argument("--mode", choices=["baseline", "vllm"], default="vllm")
    ap.add_argument("--model", default=BASELINE_MODEL,
                    help="model id to send in the vLLM payload (default: 0.5B canonical)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=128,
                    help="generation max_tokens to request (default 128)")
    ap.add_argument("--requests", type=int, default=10,
                    help="total requests to send at this concurrency (>= concurrency)")
    ap.add_argument("--stream", action="store_true",
                    help="use SSE streaming and report TTFT + ITL (vLLM/OpenAI mode only)")
    args = ap.parse_args(argv)

    if args.stream and args.mode != "vllm":
        ap.error("--stream is only supported with --mode vllm (baseline has no SSE)")

    print(f"mode={args.mode} model={args.model} url={args.url} conc={args.concurrency} "
          f"requests={args.requests} prompt_len~{len(PROMPT)} max_tokens={args.max_tokens}"
          f"{' stream' if args.stream else ''}\n"
          f"shared workload: temperature={TEMPERATURE} top_p={TOP_P}")
    r = await run_sweep(args.mode, args.url, args.concurrency, args.requests,
                        max_tokens=args.max_tokens, model=args.model, stream=args.stream)
    r["max_tokens"] = args.max_tokens
    r["model"] = args.model
    print(pretty(r))
    # JSON line for easy capture into results tables
    print("JSON\t" + json.dumps(r))
    return 0


def main() -> None:
    try:
        code = asyncio.run(amain(sys.argv[1:]))
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:  # noqa: BLE001
        print(f"bench error: {exc}", file=sys.stderr)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
