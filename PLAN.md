# PLAN.md — AI Infrastructure Engineer portfolio project: LLM serving with vLLM

A living plan and decision record. Each phase produces a portfolio artifact and
a measured result. Keep this file truthful and current: update it whenever a
phase's status or findings change. (Canonical model and hardware facts are here;
concept/arithmetic reference lives in `INFRA-CONCEPTS.md`; raw results in
`loadtest/README.md`.)

## Status at a glance

| Phase | What | Status |
|---|---|---|
| 0 | Env prep (venvs, deps, launcher, env rationale) | ✅ done |
| 1 | Serve 0.5B with vLLM; verify OpenAI endpoints | ✅ done¹ |
| 2.1 | Throughput: naive baseline vs vLLM at concurrency N | ✅ done |
| 2.2 | KV-cache/PagedAttention memory behaviour | ✅ done |
| 2.3 | Force & fix OOM (honest finding) | ✅ done² |
| 2.4 | Quantized-larger-model appendix (real weight-driven wall) | ✅ done⁴ |
| 3 | Load-testing instrumentation | ✅ done³ |
| 4 | (optional) Triton front-end | 📝 architected (docs + reference stack; deploy deferred) |
| 5 | Portfolio write-up (`README.md`, repo hygiene) | ⏳ not started |

¹ Endpoints verified on-box (curl). Host-PC browser reachability was left
  unverified and is not required for the measured results.
² Result: vLLM does **not** allocator-OOM on 0.5B — its scheduler queues instead
  of oversubscribing. A real OOM needs the quantized-larger-model appendix.
³ `loadtest/bench.py` now has a `--stream` mode reporting TTFT/ITL; `run_load.sh`
  wraps the standard set; official `vllm bench serve` provides canonical
  TTFT/ITL/throughput + Poisson `--request-rate`. Headline numbers in
  `loadtest/README.md`.
⁴ 3B-AWQ proves quantization buys KV headroom (bf16-3B won't boot); 7B-AWQ
  produces the real weight-driven wall (startup abort at max_model_len 8192;
  boots at 2048). Results in `loadtest/README.md`.

## Big picture & framing for the job hunt

The portfolio story: *"I operate the whole lifecycle that makes inference fast and
memory-safe — from hand-written PyTorch serving, through understanding why naive
serving wastes GPU memory and throughput, up to a production-grade vLLM
deployment that demonstrates continuous batching and PagedAttention, and shows how
sequence-length & KV-cache budgeting governs behaviour on an 8 GB GPU, backed by
measured load tests."*

- `main-torch.py` — the **naive but working baseline** (single-request, static KV,
  in-process). Kept deliberately for A/B comparison; **deprioritized**, not the
  centerpiece.
- vLLM (via `.venv-vllm` + `./vllm.serve.sh`) — the serving centerpiece.

## Target hardware reality (anchors every number)

- 1× RTX 3070 Laptop, **8 GB VRAM**, driver 596.08 / CUDA 13.2.
- 16 GB RAM, 16 CPUs, ~945 GB disk — single node.
- **WSL2** (`6.18.x-microsoft-standard-WSL2`). vLLM needs the WSL2/missing-toolchain
  workarounds in `vllm.env` to boot; always launch via `./vllm.serve.sh`.

## Model decision record

**Canonical serving model: `Qwen/Qwen2.5-0.5B-Instruct`** (weights cached in
`~/.cache/huggingface`). Serve this everywhere unless a phase explicitly
experiments with another tag.

Why 0.5B (and not bigger):
- The 8 GB card must hold weights **+** PagedAttention KV cache **+** activations
  **+** CUDA-graph capture at once; only ~7.0 GB is usable after driver reservation.
- **3B in bf16 (~6 GB weights)** boots but leaves **negative KV-cache space** →
  vLLM aborts (`No available memory for the cache blocks` / `Available KV cache
  memory: -0.45 GiB`). Not deployable here without quantization → not a target.
- **1.5B** fits (~2.7 GiB KV cache) but 0.5B is pinned to leave generous
  KV / sequence-length headroom across concurrent requests.
- A **quantized larger model** (e.g. 7B-AWQ) is the only route to a bigger model
  on this card; it is an optional later appendix, never the default.

> Rule for docs/config: any model field that launches the engine reads
> `Qwen/Qwen2.5-0.5B-Instruct` (or `MODEL=` override for an explicit experiment).

## Phased plan

### Phase 0 — Env prep ✅
- Two isolated venvs so the stacks don't fight: `.venv-vllm` (engine) and
  `.venv-torch` (baseline). Neither is relocatable — **never rename a venv**.
- Pins: `requirements.txt` (vLLM — the only direct dep; it vendors the rest) and
  `requirements-torch.txt` (baseline).
- Launcher + env rationale: `vllm.serve.sh` and `vllm.env` (documents the WSL2
  pin-memory/UVA, FlashInfer-sampler, CUDA-toolkit, and PATH workarounds).
- Repo hygiene: `.gitignore` excludes `.venv-*/`, caches, logs, `.opencode/`.

### Phase 1 — Serve 0.5B-Instruct with vLLM ✅
- Launch: `./vllm.serve.sh` (wraps `vllm serve Qwen/Qwen2.5-0.5B-Instruct --host
  0.0.0.0 --port 8000 --gpu-memory-utilization 0.85 --max-model-len 2048` plus the
  env overrides).
- Verified `/health`, `/v1/models`, and a sample `POST /v1/chat/completions`.
- Learned: OpenAI-compatible API surface; `gpu_memory_utilization` vs
  `max-model-len`; server-side continuous batching is active before you push load.

### Phase 2 — Pedagogical core: continuous batching + PagedAttention + OOM

1. **Throughput scaling (2.1) ✅** — identical `temperature=1.0` prompt through
   (a) baseline (single-seq, static KV) and (b) vLLM at concurrency N. vLLM
   aggregate tok/s climbs with N (~110→200→351→388 at N=1/2/4/8) while the baseline
   stays flat (~20 tok/s). Table in `loadtest/README.md`.
2. **KV-cache/PagedAttention memory (2.2) ✅** — `nvidia-smi` VRAM is ~flat
   (~7.27 GB) because vLLM **pre-allocates** the KV pool by `gpu_memory_utilization`;
   the dynamic signal is `vllm:kv_cache_usage_perc`, which rises ~linearly with
   concurrent long sequences (~0.1%→0.9% at conc 1→8). Raising `max-model-len`
   2048→8192 shrank the block budget (30081→27839 blocks; 481296→445424 tokens)
   at fixed util. Tool: `loadtest/vram_watch.py`.
3. **Force & fix OOM (2.3) ✅ — honest finding** — verified KV arithmetic
   (`2×layers×kv_heads×head_dim×dtype` = 12 KB/token for 0.5B); shrinking util to
   0.45 cut the cache to 1.9 GiB / 165,824 tokens, yet sustained 48×/128×
   concurrent bursts **stayed healthy** — vLLM queues rather than
   oversubscribing, so a true allocator-OOM is **unreachable on 0.5B**. The
   reproducible misconfiguration error is `max_tokens > max_model_len` → HTTP 400.
   Recovery levers proven: `GPU_UTIL`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`.
   A genuine OOM requires the quantized-larger-model appendix.
4. **Extra credit (optional)**: KV fragmentation (block size, partial-sequence
   waste), `--max-seq-len-to-capture`, scheduler knobs.

### Phase 2.4 — Quantized-larger-model appendix ✅ (explicit experiment)
A deliberate exception to the 0.5B-canonical rule: int4-quantized larger models to
show the **weight-driven memory wall** Phase 2.3 could not. vLLM 0.28 loads AWQ via
its bundled Marlin ops (no external quant packages, no nvcc JIT).
- **`Qwen2.5-3B-Instruct-AWQ` ✅** proves **quantization buys KV headroom**: bf16-3B
  cannot boot (negative cache); 3B-AWQ loads in 1.95 GiB with **3.99 GiB KV cache**
  (116,240 tokens) and scales tok/s with concurrency (43→153→272 at N=1/4/8).
- **`Qwen2.5-7B-Instruct-AWQ` ✅** — the **genuine wall**: 5.29 GiB weights leave
  only 0.2 GiB cache, so `max_model_len=8192` **aborts at startup** (needs 0.44 GiB);
  at 2048 it boots with a mere 1.8× concurrency. Recovery = lower `max_model_len`
  (boots, 0.98 GiB cache / 8.98×), raise util, or use 3B-AWQ.
Runs bounded (short `max_tokens`, concurrency ≤ 8). Launcher gained `QUANTIZATION`
(blank = auto-detect) and `bench.py` gained `--model`. Data in `loadtest/README.md`.
0.5B remains canonical for all other phases.

### Phase 3 — Load-testing instrumentation ✅
- `loadtest/bench.py` — async client; `--stream` reports **TTFT + ITL** (via OpenAI
  SSE) plus aggregate tok/s and e2e percentiles; `--mode baseline|vllm`, `--model`,
  `--max-tokens`. `loadtest/vram_watch.py` for occupancy/VRAM.
- `loadtest/run_load.sh` — runs the standard set (non-stream sweep, stream sweep,
  official benchmark) against a running server.
- Official `vllm bench serve` (`--backend openai-chat`) gives canonical
  TTFT/ITL/TPOT/E2E + throughput and supports Poisson `--request-rate`.
- Headline results + metric definitions: `loadtest/README.md` (Phase 3 section).
  On 0.5B: throughput scales ~121→395→635 tok/s at conc 1/4/8 while ITL stays
  ~8–9.5 ms and TTFT grows ~26→86 ms — the continuous-batching signature.
- Optional/external load tools (`hey`/`wrk`/`locust`) intentionally not used.

### Phase 4 — (optional stretch) Triton front-end ⏸→📝 architected
**Architected and documented; deployment deferred to a Docker-capable host.**
NVIDIA Triton Inference Server sits *in front of* a vLLM engine (Mode 2 front-door
proxy, or Mode 1 vLLM Python backend) and adds HTTP/gRPC/KServe, a versioned model
repository, ensembles, lifecycle control, and metrics. Full design + integration
modes + when-to-choose in `TRITON-ARCHITECTURE.md`; runnable reference stack in
`deploy/triton/`. Not deployed here because the WSL2 dev box has **no Docker**
(Triton is container-first), a native build is fragile, and the single 8 GB GPU is
already committed to vLLM — the same reasoning captured in the doc's §7.

### Phase 5 — Portfolio write-up ⏳
Fill the root `README.md` (currently empty) into one narrative:
- Motivation (naive static KV → why batched/Paged serving).
- Baseline-vs-vLLM numbers + optional graphs (`matplotlib`).
- KV-cache arithmetic + the honest OOM finding (link `INFRA-CONCEPTS.md`).
- Architecture diagram (client → WSL2 vLLM :8000; baseline :8001).
- Repo hygiene (optional restructure into `v1-baseline/`, `serve/`, `loadtest/`).

## Appendices / future work
- **Quantized-larger-model appendix** (the real OOM + weight-driven ceiling):
  a 3B-AWQ class checkpoint, weight compression vs KV headroom. Optional; carries
  WSL2/quant-toolchain risk.

## Deliverables
- Working vLLM `/v1/chat/completions` endpoint (on-box verified).
- Measured baseline-vs-vLLM throughput + KV/PagedAttention memory behaviour.
- KV arithmetic + OOM-vs-admission-control finding.
- Root `README.md` portfolio narrative (pending Phase 5).
