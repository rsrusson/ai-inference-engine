# Architecture

System overview for this project: a single-node LLM inference service on one
8 GB GPU, with a naive baseline A/B-compared against vLLM. For the detailed
serving/memory mental model see [`INFRA-CONCEPTS.md`](INFRA-CONCEPTS.md); for
measured results see [`../loadtest/README.md`](../loadtest/README.md).

## Component diagram

```
                          ┌───────────────────────────────────────────────┐
                          │                loadtest/ (client)              │
     HTTP (OpenAI API)    │  bench.py  ·  vram_watch.py  ·  run_load.sh    │
     ┌────────────────────┤  vllm bench serve (official benchmark)         │
     │                    └───────────────────────────────────────────────┘
     │                                    │
     ▼                                    ▼
┌───────────────────────────┐   ┌───────────────────────────────────────┐
│  vLLM engine  :8000       │   │  baseline server  :8001               │
│  serve/vllm.serve.sh      │   │  v1-baseline/main-torch.py            │
│                           │   │                                       │
│  continuous batching      │   │  single-request, static KV,           │
│  PagedAttention KV cache  │   │  transformers .generate() loop        │
│  OpenAI-compatible HTTP   │   │  (the "before" reference)             │
└─────────────┬─────────────┘   └───────────────────┬───────────────────┘
              │                                     │
              └───────────────┬─────────────────────┘
                              ▼
                  NVIDIA RTX 3070 Laptop, 8 GB VRAM (WSL2)
                  weights + KV cache + activations + CUDA graphs

  (optional) deploy/triton/  — NVIDIA Triton Inference Server front-end
                               (architected; needs Docker, not run on WSL2)
```

Only one server holds the GPU at a time for clean measurements.

## Request lifecycle (vLLM path)

```
client POST /v1/chat/completions {model, messages, max_tokens, temperature, ...}
  │
  ▼ tokenizer            text → token IDs
  ▼ scheduler            admit as a sequence (bounded by --max-num-seqs & free KV blocks)
  ▼ prefill              process prompt; allocate KV blocks        → determines TTFT
  ▼ decode loop          batch tokens across live sequences         → steady-state ITL
  ▼ stop                 EOS or max_tokens reached
  ▼
streamed SSE chunks (--stream) or one JSON response
```

## Where each concern lives

| Concern | Location | Notes |
|---|---|---|
| Serve the model (engine) | `serve/vllm.serve.sh`, `serve/vllm.env` | vLLM launch + WSL2 workarounds |
| Naive baseline | `v1-baseline/main-torch.py` | FastAPI + transformers, :8001 |
| Load-test clients | `loadtest/bench.py`, `vram_watch.py` | tok/s, latencies, TTFT/ITL, KV occupancy |
| Standard load set | `loadtest/run_load.sh` | sweeps + official benchmark |
| Measured results / method | `loadtest/README.md` | per-phase tables |
| Serving/memory concepts | `docs/INFRA-CONCEPTS.md` | tokens, blocks, KV math, flags |
| Roadmap & decisions | `docs/PLAN.md` | phased plan + decision record |
| Triton design (reference) | `docs/TRITON-ARCHITECTURE.md`, `deploy/triton/` | not deployed here (no Docker) |

## Scaling & memory model (summary)

- **Throughput scales with concurrency** on vLLM (continuous batching); the naive
  baseline stays flat (one sequence at a time).
- **KV cache is pre-allocated** by `--gpu-memory-utilization`; `nvidia-smi` is
  ~flat under load. Watch `vllm:kv_cache_usage_perc` on `/metrics` for real pressure.
- **KV bytes/token** = `2 × layers × kv_heads × head_dim × dtype_bytes` (12 KB for
  the 0.5B model). See `INFRA-CONCEPTS.md` §7 for the derivation.
- **Failure modes** are admission-controlled: overload queues rather than
  allocator-OOMs; a weight-driven startup abort is possible only when weights
  dominate the budget (demonstrated with a quantized 7B).

## Environments

Two isolated venvs so the serving stack and the baseline don't conflict:

| Venv | Purpose | Requirements |
|---|---|---|
| `.venv-vllm` | vLLM engine + load-test clients | `serve/requirements.txt` |
| `.venv-torch` | naive baseline (torch/transformers) | `v1-baseline/requirements-torch.txt` |
| dev/test | unit tests (no GPU) | `requirements-dev.txt` |

`make setup` / `make setup-dev` create and populate these; `make help` lists targets.
