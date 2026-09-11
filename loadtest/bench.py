#!/usr/bin/env python3
"""
bench.py — Async concurrency throughput/latency client for Phase 2.1.

Measures aggregate output tokens/sec and latency percentiles by firing
`concurrency` identical generation requests at a FastAPI/OpenAI-compatible
endpoint. Supports BOTH targets so results are directly comparable:

  * baseline  (main-torch.py :8001)  -> POST /generate   (its Request/Response schema)
  * vLLM      (:8000)                -> POST /v1/chat/completions (OpenAI schema)

Usage examples
--------------
# Baseline (single-request / static-KV reference):
python3 loadtest/bench.py --url http://127.0.0.1:8001/generate --mode baseline --concurrency 1 --requests 8

# vLLM concurrency sweep:
python3 loadtest/bench.py --url http://127.0.0.1:8000/v1/chat/completions --mode vllm --concurrency 1 --requests 15
python3 loadtest/bench.py --url http://127.0.0.1:8000/v1/chat/completions --mode vllm --concurrency 8 --requests 15

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


async def run_sweep(mode: str, url: str, concurrency: int, n_requests: int,
                    max_tokens: int = 128, model: str = BASELINE_MODEL) -> dict:
    payload = make_payload(mode, max_tokens, model)
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(concurrency)

        async def worker():
            async with sem:
                return await one_request(mode, session, url, payload)

        # Submit n_requests total, respecting concurrency cap.
        start = time.perf_counter()
        results = await asyncio.gather(*(worker() for _ in range(n_requests)))
        total_sec = time.perf_counter() - start

    ok = [r for r in results if r["ok"]]
    lats = sorted(r["lat"] for r in ok if r["lat"] is not None)
    total_out_tokens = sum(r["tokens"] for r in ok)
    errors = n_requests - len(ok)

    def pct(p):
        if not lats:
            return None
        k = max(0, min(len(lats) - 1, int(round((p / 100.0) * (len(lats) - 1)))))
        return lats[k]

    return {
        "mode": mode,
        "url": url,
        "concurrency": concurrency,
        "requests": n_requests,
        "errors": errors,
        "total_sec": total_sec,
        "total_out_tokens": total_out_tokens,
        "out_tok_s": (total_out_tokens / total_sec) if total_sec else 0.0,
        "lat_p50_ms": pct(50),
        "lat_p95_ms": pct(95),
        "lat_p99_ms": pct(99),
        "lat_mean_ms": (sum(lats) / len(lats)) if lats else None,
    }


def pretty(r: dict) -> str:
    return (
        f"[{r['mode']:8s} conc={r['concurrency']:<2d} reqs={r['requests']:<3d}] "
        f"out_tok/s={r['out_tok_s']:8.1f}  "
        f"p50={r['lat_p50_ms']:.1f}ms p95={r['lat_p95_ms']:.1f}ms p99={r['lat_p99_ms']:.1f}ms "
        f"errs={r['errors']}  total_out_tokens={r['total_out_tokens']}"
    )


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
    args = ap.parse_args(argv)

    print(f"mode={args.mode} model={args.model} url={args.url} conc={args.concurrency} "
          f"requests={args.requests} prompt_len~{len(PROMPT)} max_tokens={args.max_tokens}\n"
          f"shared workload: temperature={TEMPERATURE} top_p={TOP_P}")
    r = await run_sweep(args.mode, args.url, args.concurrency, args.requests,
                        max_tokens=args.max_tokens, model=args.model)
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
