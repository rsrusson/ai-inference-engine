# AGENTS.md

Guidance for agentic coding assistants (opencode, Claude Code, etc.) working in
this repository. Keep this file current when commands, conventions, or structure
change.

## Project overview

A portfolio/learning project demonstrating LLM inference serving on a single
small GPU: a **naive baseline** (`v1-baseline/main-torch.py`, FastAPI + transformers, static
KV) A/B-compared against a **production serving engine** (`vLLM`, continuous
batching + PagedAttention), backed by measured throughput/memory experiments.

- **Author/home**: `rsrusson` <rsrusson1@gmail.com> (repo-local git identity).
- **Primary framework**: Python 3.10, vLLM 0.28, PyTorch, FastAPI.
- **Canonical model**: `Qwen/Qwen2.5-0.5B-Instruct` on 1x RTX 3070 Laptop
  (**8 GB VRAM**) under **WSL2**. See `docs/PLAN.md` (roadmap + model decision record)
  and `docs/INFRA-CONCEPTS.md` (serving/memory cheat sheet).

> **WSL2 is the environment.** vLLM will not boot here without specific
> workarounds. **Always launch via `./serve/vllm.serve.sh`**, never a bare
> `vllm serve`, or it fails with `RuntimeError: UVA is not available` and/or
> FlashInfer JIT toolchain errors. See `vllm.env` for the rationale of each flag.

> **Docs must stay truthful.** This project is a learning artifact. A wrong claim
> (e.g. "3B is the model", "we forced a CUDA OOM") is worse than no claim. Update
> `docs/PLAN.md` and `loadtest/README.md` whenever behavior or findings change.

## Commands

### Environment setup (two separate venvs, on purpose)
```bash
# vLLM serving engine
python3 -m venv .venv-vllm
.venv-vllm/bin/pip install --upgrade pip
.venv-vllm/bin/pip install -r serve/requirements.txt

# legacy baseline (torch/transformers + FastAPI)
python3 -m venv .venv-torch
.venv-torch/bin/pip install -r v1-baseline/requirements-torch.txt
```
- **Never rename or move a venv directory.** Python venvs hard-code their absolute
  path in `bin/activate` and in every console-script shebang, so renaming breaks
  `activate` and the entry points. Recreate instead.
- Both `.venv-*/` are git-ignored. Do not commit them.

### Run — vLLM server (port 8000)
```bash
./serve/vllm.serve.sh                               # foreground; Ctrl-C to stop
nohup ./serve/vllm.serve.sh >/tmp/vllm.log 2>&1 &   # background
# Overridable: MODEL HOST PORT GPU_UTIL MAX_MODEL_LEN MAX_NUM_SEQS QUANTIZATION
#   e.g. GPU_UTIL=0.45 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=8 ./serve/vllm.serve.sh
#   quantized experiment: MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ ./serve/vllm.serve.sh
#   (QUANTIZATION blank = auto-detect from checkpoint; set only if that fails)
curl -s http://127.0.0.1:8000/health          # 200 = ready
```
- Wait for `Application startup complete.` in the log; first boot does model load,
  `torch.compile`, and CUDA-graph capture and can take ~30-60s.
- OpenAI-compatible API: `/health`, `/v1/models`, `/v1/chat/completions`,
  `/metrics`, and more (full list at `/openapi.json`).
- **Stop cleanly**: kill the `vllm serve` parent (`kill <pid>`); then verify no
  `VLLM::EngineCore` orphan remains (`pgrep -af "vllm serve|VLLM::EngineCore"`).

### Run — baseline server (port 8001)
```bash
nohup .venv-torch/bin/python v1-baseline/main-torch.py >/tmp/baseline.log 2>&1 &
curl -s http://127.0.0.1:8001/health
```
Only one side should hold the GPU at a time for clean numbers; run baseline with
vLLM stopped (or vice versa).

### Benchmarks / experiments (`loadtest/`)
```bash
# throughput at concurrency N vs vLLM (or baseline via --mode baseline)
.venv-vllm/bin/python loadtest/bench.py \
  --url http://127.0.0.1:8000/v1/chat/completions --mode vllm \
  --concurrency 8 --requests 40 --max-tokens 128
# add --model <id> when serving a non-default (e.g. AWQ) checkpoint
# add --stream for SSE TTFT + ITL (vLLM/OpenAI mode only)

# KV-cache occupancy / VRAM while forcing full-length generation
.venv-vllm/bin/python loadtest/vram_watch.py \
  --port 8000 --concurrency 8 --max-tokens 512 --requests 8 --ignore-eos

# whole standard set (non-stream sweep, stream sweep, official benchmark)
./loadtest/run_load.sh

# official canonical TTFT/ITL/throughput (Poisson with --request-rate)
.venv-vllm/bin/vllm bench serve --backend openai-chat \
  --base-url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
  --model Qwen/Qwen2.5-0.5B-Instruct --served-model-name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-name random --random-input-len 128 --random-output-len 128 \
  --num-prompts 32 --max-concurrency 8 --ignore-eos \
  --percentile-metrics ttft,itl,e2el --metric-percentiles 50,95,99 \
  --save-result --result-dir /tmp/p3_bench
```
Methodology and measured results live in `loadtest/README.md`.

### Tests / lint / typecheck
- **There is no test suite, linter, or type-checker configured.** Do not claim to
  run `pytest`/`ruff`/`mypy` unless you add and document them here.
- Quick sanity checks:
  ```bash
  .venv-vllm/bin/python -c "import ast; ast.parse(open('FILE').read())"  # py syntax
  bash -n serve/vllm.serve.sh                                             # shell syntax
  ```

## Relationship to the root README

`README.md` at the repo root is the **portfolio write-up** (Phase 5, done). It is
the public face of the repo; keep it truthful and in sync with the measured
results in `loadtest/README.md`.

## Repository layout
```
README.md               # portfolio narrative (the public-facing write-up)
AGENTS.md               # this file (kept at root for tooling discovery)
v1-baseline/
  main-torch.py         # naive baseline server (FastAPI + transformers, static KV)
  requirements-torch.txt# baseline stack (torch/transformers/fastapi pins)
  README.md             # how to run the baseline
serve/
  vllm.serve.sh         # vLLM launcher — all WSL2 env workarounds live here
  vllm.env              # sourceable env template + rationale for every workaround
  requirements.txt      # vLLM stack (vllm is the only direct dep; it vendors the rest)
docs/
  PLAN.md               # phased roadmap + model decision record
  INFRA-CONCEPTS.md     # serving/memory cheat sheet (tokens, blocks, KV math, knobs)
  TRITON-ARCHITECTURE.md# NVIDIA Triton design/integration (not deployed here)
deploy/triton/          # reference Docker stack for Triton+vLLM (needs Docker; not run)
loadtest/
  bench.py              # async concurrency client (tok/s, e2e lat; --stream TTFT/ITL)
  vram_watch.py         # KV-cache occupancy + nvidia-smi sampler
  run_load.sh           # standard load set (sweeps + official benchmark)
  README.md             # measured results + method per phase (2.1 / 2.2 / 2.3 / 2.4 / 3)
```

## Code style & conventions
- **Python 3.10**, 4-space indent. Match the existing style in `loadtest/` and
  `v1-baseline/main-torch.py`; keep files self-contained and dependency-light.
- Prefer the standard library plus what is already installed (`aiohttp`, `requests`,
  `huggingface_hub`). Do **not** add a dependency without updating the relevant `requirements*.txt`
  and this file.
- Type-hint new functions; use `from __future__ import annotations` in new scripts
  (as `loadtest/*.py` do).
- The `loadtest/` scripts print a human-readable line **and** a `JSON\t...` line so
  results are easy to capture into tables — preserve that pattern.
- Shell scripts: `set -Eeuo pipefail`, resolve paths relative to `BASH_SOURCE`
  (see `vllm.serve.sh`), and `exec` the final long-running command.
- **No comments unless they explain non-obvious rationale.** This repo documents
  *why* (env workarounds, config choices), not obvious `for` loops.

## Domain rules & gotchas
- **Canonical model is `Qwen/Qwen2.5-0.5B-Instruct`.** Do not default to 3B — it
  does not fit this GPU's KV budget (documented in `docs/PLAN.md`). Only use another
  tag for an explicitly experimental run.
- **vLLM pre-allocates its KV cache** up to `gpu_memory_utilization`, so
  `nvidia-smi` VRAM is ~flat under load. Judge memory pressure from
  `vllm:kv_cache_usage_perc` on `/metrics`, not `nvidia-smi`.
- **vLLM is admission-controlled**: under overload it queues rather than
  allocator-OOMing. A true `CUDA OOM` is **not** reachable on 0.5B here — do not
  claim to have forced one. (Phase 2.4 *does* produce a real weight-driven startup
  abort with a larger AWQ model — see `loadtest/README.md`; don't conflate the two.)
- **Quantized experiments** (Phase 2.4) are an explicit exception to the 0.5B
  canonical rule: use `MODEL=...-AWQ` (blank `QUANTIZATION` auto-detects) and
  `bench.py --model <id>`. vLLM 0.28 bundles the AWQ/Marlin ops — no extra deps.
- KV arithmetic (know these): `KV bytes/token = 2 (K&V) × layers × kv_heads ×
  head_dim × dtype_bytes`. For 0.5B = 12 KB/token; block = 16 tokens = 192 KB.
  Full derivation in `docs/INFRA-CONCEPTS.md` §5/§7.
- vLLM flag `--max-model-len` caps a single sequence's length; `--max-num-seqs`
  caps concurrent sequences; `--max-tokens` is a per-request cap. Don't conflate.

## Git
- Repo-local identity: `rsrusson` / `rsrusson1@gmail.com`; branch `main`;
  remote `origin` exists.
- Commit messages are short, descriptive, sentence/imperative style
  (e.g. `OOM test attempted/completed`, `Load testing on naive and vllm single
  instances completed`). Match that tone.
- **Only commit when explicitly asked.** Stage only intended files; never commit
  secrets, `~/.cache`, venvs, or logs.
