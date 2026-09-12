# AI Inference Engine — LLM serving on a single 8 GB GPU

[![ci](https://github.com/rsrusson/ai-inference-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/rsrusson/ai-inference-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

How **continuous batching** and **PagedAttention** make LLM inference fast and
memory-safe — demonstrated by A/B-testing a hand-written PyTorch server against
**vLLM** on one consumer GPU, with measured throughput, latency, and KV-cache
behavior.

Canonical model: `Qwen/Qwen2.5-0.5B-Instruct` · 1× RTX 3070 Laptop (8 GB) · WSL2
· vLLM 0.28 · Python 3.10.

> Every number below is measured and traced to raw results in
> [`loadtest/README.md`](loadtest/README.md). Where something could not be
> demonstrated (e.g. forcing a CUDA OOM on a tiny model), that is stated
> explicitly rather than glossed over.

---

## TL;DR — the headline result

Same model, same prompt, same sampler settings; only the serving engine differs:

| Engine | Concurrency | Output tok/s | p50 latency |
|---|---|---|---|
| Naive baseline (torch + transformers, static KV) | 1 | **~20** | ~5.0 s |
| vLLM | 1 | **110** | 0.90 s |
| vLLM | 2 | **200** | 0.91 s |
| vLLM | 4 | **351** | 1.03 s |
| vLLM | 8 | **388** | 1.00 s |

The naive server is stuck at its single-sequence rate no matter the load; vLLM's
throughput **scales with concurrency** because it continuously batches requests.
Even at concurrency 1, vLLM is ~5× faster (fused kernels + optimized decode).

Streaming metrics on vLLM show the signature of good serving — throughput rises
while **inter-token latency stays flat**:

| Concurrency | Output tok/s | TTFT p50 | ITL p50 |
|---|---|---|---|
| 1 | 121 | 26 ms | 8.1 ms |
| 4 | 395 | 43 ms | 9.3 ms |
| 8 | **635** | 86 ms | 9.5 ms |

*(Official `vllm bench serve`, conc 8: 820 tok/s, TTFT p50 64.5 ms / p99 205 ms,
ITL p50 8.7 ms / p99 15.3 ms.)*

---

## Motivation

Running an LLM is not just "forward-pass the model." A serving engine must keep
the GPU busy across many simultaneous requests while not running out of memory.
Naive serving fails both:

- **No batching** → one sequence at a time; the GPU idles between tokens.
- **Static KV cache** → each sequence reserves one large contiguous block for its
  key/value tensors, so memory is fragmented and wasted, and concurrency is
  capped hard.

vLLM (and similar engines) solve these with:

- **Continuous batching** — splice tokens from many sequences into each decode
  step, so the GPU is always fed.
- **PagedAttention** — store the KV cache in fixed-size *blocks* (pages), like an
  OS does virtual memory, enabling low-fragmentation sharing and growth.
- **KV-cache budgeting** — reserve a fixed fraction of VRAM for the cache and
  size `--max-model-len` / concurrency to fit it.

This repo measures each of those claims on real hardware.

---

## Architecture

```
        client (curl / loadtest bench.py / vllm bench serve)
                              │  HTTP (OpenAI-compatible)
                              ▼
        ┌──────────────────────────────────────────┐
        │  vLLM OpenAI server  :8000  (serve/)      │
        │    continuous batching + PagedAttention   │
        │    KV cache = gpu_memory_utilization × VRAM│
        └──────────────────────────────────────────┘
                              │ CUDA
                              ▼
                    1× RTX 3070 Laptop, 8 GB (WSL2)

        baseline (v1-baseline/main-torch.py) :8001  — naive control, run
        exclusively on the GPU for A/B comparison
```

An optional NVIDIA **Triton Inference Server** front-end is *architected but not
deployed* here (the WSL2 box has no Docker); see
[`docs/TRITON-ARCHITECTURE.md`](docs/TRITON-ARCHITECTURE.md) and
[`deploy/triton/`](deploy/triton/).

---

## Hardware & model reality

- **8 GB VRAM**, of which only ~7.0 GB is usable after driver reservation.
  Weights + KV cache + activations + CUDA-graph capture all compete for it.
- **0.5B is the canonical model.** A bf16 3B (~6 GB weights) *boots* but leaves
  **negative** KV-cache space, so vLLM refuses to serve it — documented in the
  model decision record in [`docs/PLAN.md`](docs/PLAN.md).
- **Quantization** (AWQ int4) is the lever that fits larger models (Phase 2.4).

---

## Results by phase

### Phase 2.1 — baseline vs vLLM throughput
The headline table above. vLLM scales ~110→388 tok/s at conc 1→8; the baseline
stays ~20 tok/s. Raw data: [`loadtest/README.md`](loadtest/README.md#phase-21--throughput-naive-baseline-torchtransformers-vs-vllm).

### Phase 2.2 — KV-cache / PagedAttention memory
vLLM **pre-allocates** its KV pool by `gpu_memory_utilization`, so `nvidia-smi`
stays ~flat (~7.27 GB) under load. The dynamic signal is
`vllm:kv_cache_usage_perc`, which rises with concurrent long sequences. Raising
`--max-model-len` 2048→8192 *shrank* the block budget (30,081→27,839 blocks) at
fixed utilization — the same budget, re-partitioned.

### Phase 2.3 — the honest OOM finding
On the permanent 0.5B, **vLLM does not allocator-OOM**: it is admission-controlled
and *queues* rather than oversubscribing. Deliberately shrinking the cache to
1.9 GiB and firing 48×/128× concurrent long sequences left the server healthy.
The reproducible failure is instead the config boundary:
`max_tokens > max_model_len` → HTTP 400. **A real OOM needs the next phase.**

### Phase 2.4 — quantized larger models (weight-driven memory wall, conditional)
| Model | Weight GPU mem | KV cache | Cache tokens | Verdict |
|---|---|---|---|---|
| 0.5B bf16 (canonical) | ~1 GB | ~5.5 GiB | 481,296 | fits comfortably |
| 3B bf16 | ~6 GB | negative | — | won't boot |
| **3B-AWQ int4** | 1.95 GiB | 3.99 GiB | 116,240 | fits well |
| **7B-AWQ int4** | 5.29 GiB | 0.2–0.98 GiB | 3,696–18,400 | fits, thin headroom |
| 7B-AWQ int4, `max-model-len 8192` (cold: compiles) | 5.29 GiB | 0.2 GiB | needs 0.44 | **startup abort** |
| 7B-AWQ int4, `max-model-len 8192` (warm: AOT cache) | 5.29 GiB | 0.98 GiB | 18,400 | boots (2.25×) |

- **Quantization buys headroom:** it turns an unbootable bf16 3B into a
  comfortably servable one, and makes a 7B land just inside an 8 GB card.
- **The weight-driven wall is *conditional*:** 7B-AWQ's 5.29 GiB of weights leave
  thin cache headroom, so `--max-model-len 8192` **aborts on a cold run**
  (`0.44 GiB needed > 0.2 GiB available`) but **boots once warm**. The variable is
  `torch.compile`/Inductor: cold, it compiles kernels (peak torch alloc ~1.0 GiB);
  warm, it loads the cached AOT artifact (peak ~0.26 GiB) — vLLM reserves that peak,
  so cold gets 0.2 GiB cache vs warm 0.98 GiB. (Not CUDA graphs, not serving
  activations.) Recovery: lower `max_model_len`. Lesson: reproduce in a known state
  before asserting a failure is deterministic.

### Phase 3 — load-testing instrumentation (TTFT / ITL)
Built a streaming client (`loadtest/bench.py --stream`) that measures
**time-to-first-token** and **inter-token latency**, plus the official
`vllm bench serve` for canonical numbers. The result is the continuous-batching
signature: **throughput scales, ITL stays flat, TTFT grows modestly.**

---

## Key concepts (see [`docs/INFRA-CONCEPTS.md`](docs/INFRA-CONCEPTS.md))

- **The three knobs** — `--max-model-len` (longest sequence) vs `--max-num-seqs`
  (concurrent sequences) vs `--max-tokens` (per-request output cap).
- **KV arithmetic** — `2 (K&V) × layers × kv_heads × head_dim × dtype_bytes`:
  for 0.5B that is **12 KB/token** (not the ~4 bytes of the token's text!). One
  PagedAttention block = 16 tokens = 192 KB.
- **Why 12 KB/token** — the cache stores each token's per-layer attention
  vectors, not its characters; validated against vLLM's reported ~5.5 GiB cache.
- **Reading nvidia-smi on vLLM** — it shows the pre-allocation, not cache
  fullness; use `vllm:kv_cache_usage_perc`.

---

## Reproduce

```bash
# 1) environment (two venvs: engine + baseline) — or just `make setup`
python3 -m venv .venv-vllm
.venv-vllm/bin/pip install --upgrade pip
.venv-vllm/bin/pip install -r serve/requirements.txt

python3 -m venv .venv-torch
.venv-torch/bin/pip install -r v1-baseline/requirements-torch.txt

# 2) serve 0.5B with vLLM (port 8000; WSL2 env workarounds handled inside)
nohup ./serve/vllm.serve.sh >/tmp/vllm.log 2>&1 &
curl -s http://127.0.0.1:8000/health        # 200 = ready

# 3) load-test
./loadtest/run_load.sh                       # sweeps + official benchmark

# 4) baseline (naive control) — run with vLLM stopped, port 8001
nohup .venv-torch/bin/python v1-baseline/main-torch.py >/tmp/baseline.log 2>&1 &
```

Unit tests (no GPU) run in CI and locally: `make test` (`pytest tests/`).

**Which Python runs what:** the baseline *server* uses `.venv-torch`
(torch/transformers); the vLLM *server* runs via `serve/vllm.serve.sh`
(`.venv-vllm`); and every load-test *client* (`loadtest/bench.py`,
`vram_watch.py`, `run_load.sh`) must use **`.venv-vllm`**, because those scripts
import `aiohttp` (not installed in `.venv-torch`). Don't activate `.venv-torch`
when running the client.

Notes: WSL2 + this GPU require specific vLLM env workarounds — all encapsulated
in `serve/vllm.serve.sh` and explained in `serve/vllm.env`. Never run the baseline
and vLLM simultaneously for measurements.

---

## Repository map

```
serve/         vLLM launcher (serve/vllm.serve.sh), env rationale, requirements
v1-baseline/   naive torch/transformers baseline + its own requirements
loadtest/      bench.py (tok/s, TTFT/ITL), vram_watch.py, run_load.sh, results
docs/          ARCHITECTURE.md, PLAN.md, INFRA-CONCEPTS.md, TRITON-ARCHITECTURE.md
deploy/triton/ reference Docker stack for a Triton front-end (not run here)
tests/         pytest unit tests (pure logic + API models; no GPU)
Makefile       make setup / serve / bench / test / lint (see `make help`)
```

Convenience: `make help` lists targets; `make test` runs the unit suite;
`make lint` runs the syntax checks (same as CI, see `.github/workflows/ci.yml`).

## Roadmap / status
- ✅ Phases 0–3 (env, serving, throughput/KV/OOM, load-testing)
- ✅ Phase 2.4 (quantized weight-driven wall — conditional cold/warm)
- 📝 Phase 4 (Triton) — architected, deployment deferred (needs Docker)
- See [`docs/PLAN.md`](docs/PLAN.md) for the full roadmap and decision record.

## License
MIT — see [`LICENSE`](LICENSE).
