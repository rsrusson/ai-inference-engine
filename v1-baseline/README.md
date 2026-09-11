# v1-baseline — naive serving reference (torch + transformers)

The "before" picture: a hand-written FastAPI server that loads a causal LM with
Hugging Face `transformers` and serves `/generate`. It generates **one sequence at
a time with a static KV cache and no server-side batching** — deliberately kept
as the naive control to A/B against vLLM.

> This is a *baseline*, not the project's serving engine. The production path is
> vLLM under `serve/`. See `../docs/PLAN.md` and the measured comparison in
> `../loadtest/README.md` (Phase 2.1).

## What it demonstrates
- Single-request, in-process generation (`model.generate`) with a static KV cache.
- The contrast that motivates continuous batching + PagedAttention: under
  concurrency, throughput stays ~flat (~20 tok/s for 0.5B) because requests queue
  instead of batching.

## Setup
Uses its own venv so the baseline's torch/transformers stack doesn't fight vLLM:
```bash
python3 -m venv .venv-torch
.venv-torch/bin/pip install --upgrade pip
.venv-torch/bin/pip install -r v1-baseline/requirements-torch.txt
```

## Run (binds :8001)
```bash
nohup .venv-torch/bin/python v1-baseline/main-torch.py >/tmp/baseline.log 2>&1 &
curl -s http://127.0.0.1:8001/health
```
Only one side should hold the GPU at a time for clean numbers — run the baseline
with vLLM stopped (or vice versa).

## Endpoints
- `GET  /health` — liveness + device/model info.
- `POST /generate` — body `{prompt, max_tokens, temperature, top_p}`; returns the
  completion plus `tokens_generated` and `latency_ms`.

## Model
Canonical model is `Qwen/Qwen2.5-0.5B-Instruct` (set in `main-torch.py`). See the
model decision record in `../docs/PLAN.md`.
