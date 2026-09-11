#!/usr/bin/env bash
# =============================================================================
# run_load.sh — run the standard load-test set against a running vLLM server.
#
# Assumes vLLM is already up (see ../serve/vllm.serve.sh) and reachable at BASE_URL.
# Runs three groups and prints results to stdout (also JSON lines):
#   1) non-streaming concurrency sweep      (bench.py, aggregate tok/s + e2e lat)
#   2) streaming sweep                      (bench.py --stream, TTFT + ITL)
#   3) official vllm bench serve            (canonical TTFT/ITL/throughput, JSON)
#
# Usage:
#   ./loadtest/run_load.sh                 # defaults: 0.5B on 127.0.0.1:8000
#   BASE_URL=... MODEL=... CONC_LIST="1 4 8" ./loadtest/run_load.sh
# =============================================================================
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_DIR/.venv-vllm/bin/python"
VLLM_BIN="$REPO_DIR/.venv-vllm/bin/vllm"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MAX_TOKENS="${MAX_TOKENS:-128}"
CONC_LIST="${CONC_LIST:-1 4 8}"
RESULT_DIR="${RESULT_DIR:-/tmp/p3_bench}"

CHAT_URL="$BASE_URL/v1/chat/completions"

echo "== run_load.sh =="
echo "base_url=$BASE_URL model=$MODEL max_tokens=$MAX_TOKENS conc='$CONC_LIST'"
echo "health: $(curl -s -o /dev/null -w '%{http_code}' "$BASE_URL/health" || echo FAIL)"
echo

echo "--- 1) non-streaming concurrency sweep ---"
for N in $CONC_LIST; do
    "$PY" "$REPO_DIR/loadtest/bench.py" --url "$CHAT_URL" --mode vllm \
        --model "$MODEL" --concurrency "$N" --requests "$((N * 2))" --max-tokens "$MAX_TOKENS"
done
echo

echo "--- 2) streaming sweep (TTFT + ITL) ---"
for N in $CONC_LIST; do
    "$PY" "$REPO_DIR/loadtest/bench.py" --url "$CHAT_URL" --mode vllm --stream \
        --model "$MODEL" --concurrency "$N" --requests "$((N * 2))" --max-tokens "$MAX_TOKENS"
done
echo

echo "--- 3) official vllm bench serve ---"
mkdir -p "$RESULT_DIR"
"$VLLM_BIN" bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL" \
    --endpoint /v1/chat/completions \
    --model "$MODEL" --served-model-name "$MODEL" \
    --dataset-name random --random-input-len 128 --random-output-len "$MAX_TOKENS" \
    --num-prompts 32 --max-concurrency 8 --ignore-eos \
    --percentile-metrics ttft,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir "$RESULT_DIR"
echo
echo "official results saved under: $RESULT_DIR"
