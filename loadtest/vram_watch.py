#!/usr/bin/env python3
"""
vram_watch.py — Phase 2.2 / 2.3: watch PagedAttention KV-cache fill + VRAM during load.

vLLM PRE-ALLOCATES its KV cache at startup (up to gpu_memory_utilization x VRAM),
so raw `nvidia-smi` used-memory looks ~flat across concurrency. The dynamic,
meaningful signal is vLLM's cache fill ratio, `vllm:kv_cache_usage_perc`
(exposed on the server's /metrics). It rises toward 1.0 as concurrent long
sequences consume PagedAttention blocks.

This script:
  1. Fires `concurrency` identical /v1/chat/completions requests (large max_tokens
     so sequences are long enough to genuinely fill cache blocks).
  2. While they are in flight, samples `vllm:kv_cache_usage_perc` from /metrics
     AND `nvidia-smi` used-memory at ~3 Hz.
  3. Reports: peak cache-usage %, peak GPU mem, nvidia-smi idle baseline,
     per-request stats, and the number of GPU cache blocks (from cache_config_info).

Usage:
  python3 loadtest/vram_watch.py --port 8000 --concurrency 2 --max-tokens 512 --requests 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time

import aiohttp

PROMPT = "Explain what an inference-serving engine does in two or three short sentences."
TEMPERATURE = 1.0
TOP_P = 0.9
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

_TYPE_CACHE_USAGE = re.compile(
    r'^vllm:kv_cache_usage_perc\{[^}]*model_name="([^"]+)"[^}]*\} ([0-9.eE+-]+)', re.M
)
_TYPE_CACHE_CONFIG = re.compile(r'^vllm:cache_config_info\{.*\} ([0-9.eE+-]+)', re.M)
_BLOCK_RE = re.compile(r'num_gpu_blocks="(\d+)"')


async def get_cache_usage(session: aiohttp.ClientSession, metrics_url: str) -> float | None:
    try:
        async with session.get(metrics_url) as r:
            text = await r.text()
    except Exception:
        return None
    for m in _TYPE_CACHE_USAGE.finditer(text):
        return float(m.group(2))
    return None


async def get_cache_blocks_count(session, metrics_url: str) -> dict | None:
    try:
        async with session.get(metrics_url) as r:
            text = await r.text()
    except Exception:
        return None
    # cache_config_info has a `num_gpu_blocks="NNN"` label
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info"):
            mm = _BLOCK_RE.search(line)
            if mm:
                return {"num_gpu_blocks": int(mm.group(1)),
                        "gpu_memory_utilization": _label(line, "gpu_memory_utilization"),
                        "block_size": _label(line, "block_size"),
                        "kv_cache_size_tokens": _label(line, "kv_cache_size_tokens")}
    return None


def _label(line: str, key: str) -> str | None:
    m = re.search(rf'{key}="([^"]*)"', line)
    return m.group(1) if m else None


def nvidia_used_mb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return float(out.splitlines()[0])
    except Exception:
        return None


async def one_request(mode_url: str, session, conc_index: int, max_tokens: int,
                      ignore_eos: bool):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "ignore_eos": ignore_eos,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(mode_url, json=payload) as r:
            body = await r.json()
            status = r.status
        out_tokens = int((body.get("usage") or {}).get("completion_tokens", 0))
    except Exception as exc:
        return {"ok": False, "err": repr(exc), "lat_ms": None, "tok": 0}
    return {"ok": status == 200, "err": None, "lat_ms": (time.perf_counter() - t0) * 1000.0,
            "tok": out_tokens}


async def amain(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=512)
    ap.add_argument("--requests", type=int, default=12,
                    help="total requests to send (>= concurrency)")
    ap.add_argument("--ignore-eos", dest="ignore_eos", action="store_true",
                    help="generate the full max_tokens length (ignore EOS) so "
                         "sequences stay resident and fill the KV cache")
    args = ap.parse_args(argv)

    base = f"http://{args.host}:{args.port}"
    gen_url = f"{base}/v1/chat/completions"
    metrics_url = f"{base}/metrics"

    async with aiohttp.ClientSession() as s:
        blocks = await get_cache_blocks_count(s, metrics_url)
        idle_usage = await get_cache_usage(s, metrics_url)
    idle_smi = nvidia_used_mb()

    print("== warm/idle baseline ==")
    print(f"  vllm kv_cache_usage_perc@idle = {idle_usage}")
    print(f"  nvidia-smi used (idle)        = {idle_smi} MiB")
    if blocks:
        print(f"  cache blocks configured       = {blocks}")

    async with aiohttp.ClientSession() as s:
        loops = int(max(1, args.requests / args.concurrency))

        sam = {"peak_usage": 0.0, "peak_smi": 0.0, "n": 0}

        async def sample():
            while True:
                u = await get_cache_usage(s, metrics_url)
                smi = nvidia_used_mb()
                if u is not None:
                    sam["peak_usage"] = max(sam["peak_usage"], u)
                    sam["n"] += 1
                if smi is not None:
                    sam["peak_smi"] = max(sam["peak_smi"], smi)
                await asyncio.sleep(0.3)

        def make_gen():
            # one batch of `concurrency` requests
            async def _go():
                return await asyncio.gather(
                    *(one_request(gen_url, s, i, args.max_tokens, args.ignore_eos)
                      for i in range(args.concurrency))
                )

            return _go()

        results = []
        sampler = asyncio.create_task(sample())
        t_load = time.perf_counter()
        for _ in range(loops):
            results += await make_gen()
        load_sec = time.perf_counter() - t_load
        await asyncio.sleep(0.4)
        sampler.cancel()

    ok = [r for r in results if r["ok"]]
    lats = sorted(r["lat_ms"] for r in ok if r["lat_ms"] is not None)
    out_tok = sum(r["tok"] for r in ok)
    errs = len(results) - len(ok)

    def pct(p):
        if not lats:
            return None
        k = max(0, min(len(lats) - 1, round((p / 100.0) * (len(lats) - 1))))
        return lats[k]

    idle_gb = None
    if sam["peak_smi"]:
        peak_gb = sam["peak_smi"] / 1024.0
        idle_gb = (idle_smi / 1024.0) if idle_smi else None

    print("\n== under load ==")
    print(f"  concurrency={args.concurrency}  requests={len(results)}  max_tokens={args.max_tokens}")
    print(f"  PEAK kv_cache_usage_perc = {sam['peak_usage']:.3f}  (sampled {sam['n']}x)")
    print(f"  nvidia peak used / idle  = {peak_gb:.2f} GiB / {idle_gb:.2f} GiB  ({sam['peak_smi']:.0f} MiB pk)")
    print(f"  p50={pct(50):.0f}ms p95={pct(95):.0f}ms  out_tokens={out_tok}  errors={errs}")
    summary = {
        "concurrency": args.concurrency, "requests": len(results),
        "max_tokens": args.max_tokens, "peak_kv_usage_perc": sam["peak_usage"],
        "peak_vram_mib": sam["peak_smi"], "idle_vram_mib": idle_smi,
        "out_tokens": out_tok, "errors": errs, "load_sec": round(load_sec, 2),
        "out_tok_s": round(out_tok / load_sec, 1) if load_sec else None,
        "p50_ms": pct(50), "p95_ms": pct(95), "cache": blocks,
    }
    print("JSON\t" + json.dumps(summary))
    return 0


def main() -> None:
    try:
        code = asyncio.run(amain(sys.argv[1:]))
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:  # noqa: BLE001
        print(f"vram_watch error: {exc}", file=sys.stderr)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
