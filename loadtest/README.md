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

---

# Phase 2.2 — KV cache / PagedAttention memory behaviour

Model: `Qwen/Qwen2.5-0.5B-Instruct`, GPU util 0.85, `vllm.serve.sh`. Drives
`vram_watch.py`, which reports three things: the **idle(config)** state via
`/metrics`, the **peak `kv_cache_usage_perc`** sampled ~3 Hz while requests are in
flight, and `nvidia-smi` used-memory.

## Key architectural finding first (important nuance)

vLLM does **NOT** grow its cache memory with load in the naive sense: it
**pre-allocates** a PagedAttention block pool at startup, sized by
`gpu_memory_utilization` (0.85 x usable VRAM). So total GPU memory held is ~flat
once warm, regardless of concurrency:

  nvidia-smi used (idle)  ~= 7.27 GB  and stays ~equal under full load.

What actually changes with load is **how much of that pre-allocated pool is
consumed by KV blocks** — exposed as `vllm:kv_cache_usage_perc` on `/metrics`.
This is the dynamic signal that "KV grows as sequences pile up."

## A) KV-cache occupancy vs concurrency (max-model-len 2048, forced length)

Shorter answers (real chat) free blocks quickly, so to *see* blocks accumulate we
force full-length output (`ignore_eos: true`, max_tokens=512) so live sequences
hold their KV blocks while piling up.

| Concurrent full-length seqs | Peak kv_cache_usage_perc | out tokens | total_sec |
|---|---|---|---|
| 1  | ~0.1 % |  512 | 4.9 |
| 4  | ~0.4 % | 2048 | 5.5 |
| 8  | ~0.9 % | 4096 | 9.3 |

Block-usage scales ~linearly with concurrent sequences (each active sequence owns
its KV blocks until it finishes) — the PagedAttention pool is drained by live
requests exactly as predicted.

Why the *percentages* are tiny: 0.5B has few layers / KV heads, so even an 8-way
burst of 512-token sequences touches <1% of a cache that is sized as ~481k tokens
`(481296)` `num_gpu_blocks=30081` @ max-len 2048. On this small model the cache is
vastly over-provisioned relative to realistic load — a deliberate headroom choice
for the concurrency story (see model note). Memory-pressure/OOM can't be meaningfully
reached here *unless* the cache budget is shrunk (lower util) or `max_model_len`
is raised to shrink per-sequence fit — which is exactly the Phase 2.3 experiment.

## B) Effect of raising --max-model-len (same GPU util 0.85)

Re-booted with `MAX_MODEL_LEN=8192`. With `gpu_memory_utilization` FIXED, total
cache memory stays ~constant; what changes is how the pool and graph-capture are
provisioned:

| Metric | max-model-len 2048 | max-model-len 8192 | effect |
|---|---|---|---|
| kv_cache_size_tokens | 481296            | 445424             | less token capacity |
| num_gpu_blocks       | 30081              | 27839              | fewer PagedAttention blocks |
| kv_cache_max_concurrency | higher        | 54.4               | fewer (longest) concurrent seqs fit before OOM |
| idling  used (nvidia-smi) | ~7.27 GB    | ~7.27 GB           | ~flat (util-bound, not length-bound) |

Interpretation: raising max-model-len does NOT add memory; it re-budgets the same
reserved pool AND increases the largest CUDA-graph capture, so you end with
**fewer** blocks/tokens of headroom and lower max concurrency per sequence length.
It is the configuration lever that governs per-sequence headroom; combined with a
shrunk util budget it is the axis explored for OOM in Phase 2.3 (see below).

## Reproduce (Phase 2.2)
```bash
# vLLM @ max-model-len 2048 (default)
nohup ./vllm.serve.sh >/tmp/vllm.log 2>&1 &
python3 loadtest/vram_watch.py --port 8000 --concurrency 1 --max-tokens 512 --requests 1 --ignore-eos
python3 loadtest/vram_watch.py --port 8000 --concurrency 4 --max-tokens 512 --requests 4 --ignore-eos
python3 loadtest/vram_watch.py --port 8000 --concurrency 8 --max-tokens 512 --requests 8 --ignore-eos

# vLLM @ max-model-len 8192 to compare config/headroom
MAX_MODEL_LEN=8192 nohup ./vllm.serve.sh >/tmp/vllm8192.log 2>&1 &
curl -s http://127.0.0.1:8000/metrics | grep -a cache_config_info
```

---

# Phase 2.3 — OOM forcing attempt & the honest finding

Attempted to force a `CUDA out of memory` on the permanent 0.5B model (Option B:
shrink the reserved KV pool, then hammer it with long concurrent sequences). The
result is more interesting than a crash: **0.5B will not allocator-OOM on this
card.** This section records the KV arithmetic, the admission-control behaviour
proved, and why a real OOM needs the quantized-larger-model appendix.

## KV-cache arithmetic (validated against vLLM)

`Qwen2.5-0.5B` / Qwen2 arch config:
- `num_hidden_layers` = 24, `num_key_value_heads` = 2 (GQA, head_dim 64), dtype bf16 (2 B).

KV bytes per stored token = `2 (K&V) × layers × kv_heads × head_dim × dtype_bytes`
= `2 × 24 × 2 × 64 × 2 = 12,288 B ≈ 12 KB / token`.

vLLM reports cache capacity in tokens and this matches: at max_len.2048 util .85,
`kv_cache_size_tokens`=481296 → 481296 × 12288 B ≈ 5.5 GiB cache ≈ the reported
`Available KV cache memory: 5.1–5.5 GiB`. Formula checks out.

## We CAN shrink the cache; we CANNOT make it run out

Under-provision by booting with low GPU util (+ max-model-len 4096):

| Config (MAX_MODEL_LEN / GPU_UTIL) | cache size tokens | GPU KV cache | num_gpu_blocks | nvidia-smi idle |
|---|---|---|---|---|
| 2048 / 0.85 (default) | 481296 | ~5.5 GiB | 30081 | ~7.27 GB |
| 4096 / 0.45 | 165824 | 1.9 GiB | 10364 | ~4.00 GB  |

Then sustained bursts against the 1.9 GiB cache (util .45, max-len 4096):

| Load | kv usage peak | out tok/s | errors | server state |
|---|---|---|---|---|
| 48 concurrent × 1500 tok | 44 % | 2497 | 0 | healthy (p50 ~28 s) |
| 128 concurrent × 3500 tok × 256 | — | — | 0 | **still healthy** (200) |

**Why no OOM:** vLLM's scheduler is *admission controlled*. It never admits more
in-flight sequences than the pre-allocated block pool can hold; excess demand is
**queued**, not allocated. And 0.5B's weights+activations fit easily even in 45%
of the card, so no genuinely unmeetable allocation ever occurs. The observable cost
of overload is latency/queueing growth, not a crash.

## The reproducible misconfiguration error (config lever does bite)
What DOES deterministically reject is requesting more output than the engine allows:
```
max_tokens=5000 cannot be greater than max_model_len=4096  (HTTP 400, BadRequest)
```
This is the same **`--max-model-len` lever** from the fix story: set it too low for
the deployment's prompt+output length and valid requests bounce with a clear error;
raise it (subject to VRAM/KV arithmetic) and they pass.

## Conclusion / why a real OOM needs the appendix
A true raster-level `CUDA out of memory` on permanent 0.5B is effectively
unreachable, because (a) the model is far too small to starve even a shrunk cache,
and (b) the memory manager refuses to oversubscribe. The genuinely dramatic OOM —
weights that genuinely do not fit → startup abort (`No available memory for the
cache blocks`, as seen with 3B) or runtime exhaustion — requires the
**quantized-larger-model appendix** (e.g. 3B-AWQ, or a very-low-util + very-large
max-len pathological config), which compresses weights enough to boot but still
far closer to the memory ceiling. That is deliberately pending as an optional later
milestone; the value captured here is the admission-control + KV-arithmetic lesson,
which is the transferable infra concept.

## Recovery levers exercised (so they are documented where they do matter)
- `GPU_UTIL` — right-sizes the reserved KV pool to workload (0.45 → drops idle VRAM
  7.3→4.0 GB, shrinks 30081→10364 blocks; too-low starves cache for long seqs).
- `MAX_MODEL_LEN` — bounds per-sequence length & graph capture; too-low rejects long
  prompts via 400, too-high shrinks blocks/headroom (see Phase 2.2 §B).
- `MAX_NUM_SEQS` — optional admission cap (added to `vllm.serve.sh` as `MAX_NUM_SEQS`);
  enforces a hard bound on simultaneous sequences to keep per-request latency bounded
  under heavy traffic instead of unbounded queueing.
- `vram_watch.py --max-tokens N --ignore-eos` — keeps sequences resident long enough to
  actually observe cache occupancy (short chat answers free blocks too fast to study).

## Reproduce
```bash
# under-provisioned cache (util 0.45, longer max len)
GPU_UTIL=0.45 MAX_MODEL_LEN=4096 nohup ./vllm.serve.sh >/tmp/vllm45.log 2>&1 &
curl -s http://127.0.0.1:8000/metrics | grep -a cache_config_info   # blocks/cache size
# sustained burst -> queues, does NOT OOM:
python3 loadtest/vram_watch.py --port 8000 --concurrency 48 --max-tokens 1500 --requests 48 --ignore-eos
# deterministic config rejection -> fix lever:
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":5000}'

# optional admission cap on top:
GPU_UTIL=0.45 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=8 nohup ./vllm.serve.sh >/tmp/vllm_cap.log 2>&1 &
```

---

# Phase 2.4 — Quantized-larger-model appendix (weight-driven wall)

Explicit experiment against `Qwen2.5-3B-Instruct-AWQ` and `Qwen2.5-7B-Instruct-AWQ`
both **AWQ int4** (`quant_method=awq`, group 128) to show the memory wall that a
0.5B model cannot reach, and to prove **quantization buys KV headroom**. vLLM 0.28
loads these with its bundled ops — Marlin int4 kernel on sm_86 (RTX 3070), **no
`autoawq` package and no nvcc/JIT** required. Auto-detection picked `auto_awq`.

## The quantization lever: 3B bf16 (won't boot) vs 3B-AWQ (boots) ✅

| 3B variant | Weights on GPU | Boots? | KV cache available | cache tokens | max concurrency @2048 |
|---|---|---|---|---|---|
| bf16 (`Qwen2.5-3B-Instruct`) | ~6 GB | ❌ abort | negative (−0.45 GiB) | — | — |
| **AWQ int4** | **1.95 GiB** | ✅ | **3.99 GiB** | 116,240 | 56.8× |

- `Model loading took 1.95 GiB` (int4) vs ~6 GB bf16 → **quantization is what makes
  a 3B-class model servable on 8 GB at all.** This is the practical counterpart to
  the "fp16→fp8 halves KV" note in `INFRA-CONCEPTS.md` §7.
- Idle `nvidia-smi`: **7385 MiB** (util 0.85); `num_gpu_blocks=7265`, block 16.

### Bounded throughput (3B-AWQ, max_tokens=64)
| Concurrency | out tok/s | p50 | notes |
|---|---|---|---|
| 1 | ~43 | 0.61 s | |
| 4 | ~153 | 0.83 s | |
| 8 (cold) | ~47 | 5.81 s | first batch-8: compile/cudagraph warmup artifact |
| 8 (warm) | **272** | 0.96 s | re-run; scaling holds |

Scaling mirrors Phase 2.1 (aggregate tok/s rises with N). The first N=8 run is a
**cold-start** effect, not a steady-state regression. Quality check: an int4 sample
answer about PagedAttention was coherent and on-topic, so int4 remains usable.

## The genuine weight-driven wall: 7B-AWQ ✅ (what 2.3 could not produce)

7B-AWQ weights are **5.29 GiB** on the 8 GB card. Its KV arithmetic is also heavier:
head_dim 128, 28 layers, 4 KV heads → `2×28×4×128×2 = 56 KB/token`
(vs 12 KB/token for 0.5B).

| 7B-AWQ config | GPU KV cache | cache tokens | max concurrency @2048 | result |
|---|---|---|---|---|
| `--max-model-len 2048` | 0.2 GiB | 3,696 | **1.8×** | boots, barely (1-2 concurrent seqs) |
| `--max-model-len 8192` | needs 0.44 GiB, only 0.2 GiB free | — | — | **hard startup abort** |

The `max_model_len=8192` startup failed with the real memory-arithmetic wall:

```
ValueError: To serve at least one request with the model's max seq len (8192),
0.44 GiB KV cache is needed, which is larger than the available KV cache memory (0.2 GiB).
Based on the available memory, the estimated maximum model length is 3696.
Try increasing `gpu_memory_utilization` ... or decreasing `max_model_len` ...
```

This is the weight-driven ceiling in concrete form: 5.29 GiB weights leave only
~0.2 GiB for PagedAttention, so vLLM **refuses to start** rather than oversubscribe.
Unlike Phase 2.3 (load-time queueing on 0.5B), this is a genuine, reproducible
**OOM-class failure** — the engine's own message states the max supportable length
(3696) and the levers.

### Recovery lever ✅
Relaunched at `--max-model-len 2048` (a length the weight-crowded budget can hold):
boots cleanly, **0.98 GiB cache / 18,400 tokens / 8.98× concurrency**, `/health` 200,
serves coherent int4 output. So the fix story is:
**lower `max_model_len` to fit the post-weights budget** (alternatives: raise
`gpu_memory_utilization`, or use the smaller 3B-AWQ).

## Consolidated weight-vs-KV table (all measured)
| Model (mode) | Weight GPU mem | KV cache avail | cache tokens | verdict |
|---|---|---|---|---|
| 0.5B bf16 (canonical) | ~1 GB | ~5.5 GiB | 481,296 | fits comfortably |
| 3B bf16 | ~6 GB | negative | — | **won't boot** |
| 3B-AWQ int4 | 1.95 GiB | 3.99 GiB | 116,240 | fits well |
| 7B-AWQ int4 @2048 | 5.29 GiB | 0.2 GiB | 3,696 | fits w/ ~1.8× concurrency |
| 7B-AWQ int4 @8192 | 5.29 GiB | needs 0.44 GiB | — | **startup abort** |

## Takeaways
- **Quantization is a memory lever**: it converts an unbootable 3B into a
  comfortably servable one, and makes a 7B land *just* within an 8 GB card.
- **Weights and KV compete for the same budget**: once weights dominate (7B), KV
  collapses and `max_model_len` must shrink — the exact trade the plan wanted to show.
- **The docs rule still holds**: 0.5B remains the canonical model; this phase is a
  deliberate, bounded experiment using the launcher's `MODEL=` / `QUANTIZATION=`
  overrides. No new Python deps were needed (ops are bundled in vLLM).

## Reproduce (Phase 2.4)
```bash
# 3B-AWQ: quantization makes a 3B servable (bf16 3B won't boot)
MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ nohup ./vllm.serve.sh >/tmp/vllm_3bawq.log 2>&1 &
.venv-vllm/bin/python loadtest/bench.py --url http://127.0.0.1:8000/v1/chat/completions \
  --mode vllm --model Qwen/Qwen2.5-3B-Instruct-AWQ --concurrency 8 --requests 8 --max-tokens 64

# 7B-AWQ: weight-driven wall (startup abort at 8192)
MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ MAX_MODEL_LEN=8192 nohup ./vllm.serve.sh >/tmp/vllm_7b_8k.log 2>&1 &
# ... then the recovery: a max_len the weights leave room for
MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ MAX_MODEL_LEN=2048 nohup ./vllm.serve.sh >/tmp/vllm_7b.log 2>&1 &
```
(Optional explicit flag if auto-detect ever fails: `QUANTIZATION=awq_marlin`.)
