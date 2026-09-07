#!/usr/bin/env bash
# =============================================================================
# vllm.serve.sh — Launch the vLLM OpenAI-compatible inference server.
#
# Captures the WSL2 + missing-CUDA-toolkit workarounds required to boot vLLM
# 0.28 in THIS environment (see vllm.env for the rationale):
#
#   * VLLM_WSL2_ENABLE_PIN_MEMORY=1
#       vLLM disables pinned memory (and therefore CUDA UVA, used by its buffer
#       allocator) when it detects WSL2. Without this you get:
#         RuntimeError: UVA is not available
#
#   * VLLM_USE_FLASHINFER_SAMPLER=0
#       FlashInfer's top-k/top-p sampler is JIT-compiled with nvcc. We have no
#       matching system CUDA toolkit here, only the runtime driver + a mismatched
#       pip nvcc, so the JIT build fails. Disabling it makes vLLM use its native
#       Triton sampler (functionally equivalent, no recompile).
#
#   * CUDA_HOME -> pip-installed nvcc (nvidia/cu13)
#       Lets tooling that probes $CUDA_HOME/bin/nvcc find a compiler. Only needed
#       if you keep FlashInfer features that JIT; harmless to keep set.
#
#   * PATH
#       Prepend the venv bin so tools like `ninja` (used during JIT builds) are
#       discoverable on subprocess PATH.
#
# Usage:
#   ./vllm.serve.sh                        # run in foreground, default model
#   MODEL=... PORT=... ./vllm.serve.sh     # override model/port
#   ./vllm.serve.sh --log /tmp/vllm.log &  # run detached, tee logs to a file
# =============================================================================
set -Eeuo pipefail

# --- Repo bootstrap (resolve paths regardless of CWD) -------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_VLLM="$REPO_DIR/.venv-vllm"
VLLM_BIN="$VENV_VLLM/bin/vllm"
CUDA13_HOME="$VENV_VLLM/lib/python3.10/site-packages/nvidia/cu13"

# --- Overridable defaults -----------------------------------------------------
export MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export GPU_UTIL="${GPU_UTIL:-0.85}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
# Optional admission cap (blank => let vLLM auto-tune). Set e.g. 16 to enforce a
# hard limit on simultaneously-running sequences (a recovery lever for 2.3).
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"

# --- Fixed environment (see header) -------------------------------------------
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="$CUDA13_HOME"
export PATH="$VENV_VLLM/bin:$PATH"

# --- Execute ----------------------------------------------------------------
if [[ ! -x "$VLLM_BIN" ]]; then
    echo "[vllm.serve] ERROR: '$VLLM_BIN' not found." >&2
    echo "[vllm.serve] Did you run:  python3 -m venv .venv-vllm && .venv-vllm/bin/pip install -r requirements.txt" >&2
    exit 1
fi

echo "[vllm.serve] model=${MODEL} host=${HOST} port=${PORT} mem_util=${GPU_UTIL} max_len=${MAX_MODEL_LEN}${MAX_NUM_SEQS:+ max_num_seqs=${MAX_NUM_SEQS}}"
echo "[vllm.serve] CUDA_HOME=${CUDA_HOME}"

ARGS=(serve "$MODEL" --host "$HOST" --port "$PORT"
      --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_MODEL_LEN")
if [[ -n "$MAX_NUM_SEQS" ]]; then
    ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

exec "$VLLM_BIN" "${ARGS[@]}"
