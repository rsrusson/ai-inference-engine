# Phase 2.1 — Throughput: naive baseline (torch/transformers) vs vLLM

Measured on this box (RTX 3070 Laptop, 8 GB, WSL2). Model used everywhere:
**`Qwen/Qwen2.5-0.5B-Instruct`** (fp16/bf16), canonical for this project.

## Model note (canonical / decision record)
The permanent serving model is **0.5B-Instruct**, NOT 3B:
- This 8 GB card must hold weights + KV cache + activations + CUDA-graph memory.
  Only ~7.0 GB is usable after driver reservation.
- **3B in bf16 (~6 GB weights)** leaves **negative KV-cache space** → vLLM refuses
  to boot: `No available memory for the cache blocks` / `Available KV cache: -0.45 GiB`.
  It is not deployable here without quantization, so it is not part of the core plan.
- **1.5B** fits (~2.7 GiB KV) but we standardize on **0.5B** for larger KV/sequence
  headroom under concurrency (cached KV = **5.1 GiB** here).
- A quantized 7B-AWQ is the only path to a bigger model on this card; optional appendix.

Any tracked config/launcher that selects a model MUST say `Qwen/Qwen2.5-0.5B-Instruct`
unless a run is explicitly experimenting with another tag.

## Method (identical workload each side)
- Same prompt (~78 tokens): *"Explain what an inference-serving engine does in two or
  three short sentences."*
- Same sampler settings: `temperature=1.0`, `top_p=0.9`, `max_tokens=128`.
- **baseline** = `main-torch.py` (`POST /generate`, transformers + torch fp16, static KV,
  in-process generation, **no dynamic batching**). Measured at its natural concurrency = 1
  (it can only meaningfully serve one sequence; higher N just queues).
- **vLLM** = `vllm serve` (`POST /v1/chat/completions`). Continuously batches concurrent
  requests (PagedAttention). Swept over N ∈ {1,2,4,8} concurrent in-flight requests.
- Both capture aggregate output `tok/s` (generated tokens / wall time) + latency p50/p95.
- Servers ran at different times on an exclusive GPU to avoid contention.

## Results
| Engine | Concurrency N | Output tok/s | p50 (s) | p95 (s) | Errors |
|---|---|---|---|---|---|
| baseline (torch, static KV) | 1 | **~20** | ~5.0 | ~6.6 | 0 |
| vLLM | 1 | **110** | 0.90 | 1.16 | 0 |
| vLLM | 2 | **200** | 0.91 | 0.99 | 0 |
| vLLM | 4 | **351** | 1.03 | 1.29 | 0 |
| vLLM | 8 | **388** | 1.00 | 5.42 | 0 |

Baseline is avg of two 6-request runs (~18.5 and 21.4 tok/s). vLLM rows at N in
{1,2,4,8} used {6,12,24,40} total requests each. 0 errors across all runs.

### What this shows (the point)
- The **naive baseline is stuck ~20 tok/s** regardless of load — every request waits on
  the last one; the GPU is underutilized and there is no server-side batching.
- **vLLM scales throughput with concurrency**: ~110 → 200 → 351 → 388 tok/s as N rises.
  Continuous batching keeps the GPU fed by interleaving multiple sequences between
  decode steps (PagedAttention).
- Even **vLLM at N=1 (~110 tok/s) is ~5× the baseline**: fused kernels + optimized
  decode beat the naive per-token transformers loop.
- Throughput growth flattens by N=8 and **p95 latency degrades** (5.4 s) — the GPU is
  nearing saturation and a 40-request burst at once queues. Classic saturating curve.

## Notes / caveats
- tok/s here are **aggregate output tokens/sec** across all requests, a fair cross-engine
  measure (same model + workload).
- VRAM: vLLM (0.5B, util 0.85) holds ~7.27 GiB total once warm — of which ~5.1 GiB is
  the **pre-allocated PagedAttention KV cache**. vLLM reserves its cache budget up front
  (Phase 2.2 digs into VRAM-vs-concurrency/max-length).
- Numbers are single-run, single-node, 0.5B; treat as indicative of the *mechanism*, not
  an appliance benchmark. See `bench.py` to re-run:
  `python3 loadtest/bench.py --help`

## Reproduce
```bash
# baseline (needs .venv-torch rebuilt; binds :8001)
nohup .venv-torch/bin/python main-torch.py >/tmp/baseline.log 2>&1 &
python3 loadtest/bench.py --url http://127.0.0.1:8001/generate --mode baseline --concurrency 1 --requests 6

# vLLM (binds :8000, defaults to 0.5B)
nohup ./vllm.serve.sh >/tmp/vllm.log 2>&1 &
python3 loadtest/bench.py --url http://127.0.0.1:8000/v1/chat/completions --mode vllm --concurrency 1  --requests 6
python3 loadtest/bench.py --url http://127.0.0.1:8000/v1/chat/completions --mode vllm --concurrency 2  --requests 12
python3 loadtest/bench.py --url http://127.0.0.1:8000/v1/chat/completions --mode vllm --concurrency 4  --requests 24
python3 loadtest/bench.py --url http://127.0.0.1:8000/v1/chat/completions --mode vllm --concurrency 8  --requests 40
```
GPU is exclusive per side; do baseline first, then vLLM (or vice versa), not together.
