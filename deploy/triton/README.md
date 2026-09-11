# deploy/triton — reference stack (NVIDIA Triton Inference Server + vLLM)

**Reference only — not run on the dev box.** The WSL2 dev machine has no Docker;
see `../../TRITON-ARCHITECTURE.md` §7 for the decision record. This directory is
a template to run on a Docker-capable host with an NVIDIA GPU.

## What it does
Implements **Mode 2** from `TRITON-ARCHITECTURE.md`: Triton runs a
Python-backend **proxy model** (`llm_proxy`) that forwards to a vLLM OpenAI
server. vLLM stays exactly as this repo runs it; Triton adds the
HTTP/gRPC/KServe/metrics surface in front.

## Prerequisites
- Docker + Docker Compose
- NVIDIA Container Toolkit (so the container can see the GPU)
- A running vLLM server (this repo: `../../vllm.serve.sh`) reachable from Triton

## Run
```bash
# 1) start vLLM (on the host)
cd ../.. && nohup ./vllm.serve.sh >/tmp/vllm.log 2>&1 &
curl -s http://127.0.0.1:8000/health          # 200 = ready

# 2) start Triton (front door)
cd deploy/triton
VLLM_URL=http://127.0.0.1:8000 docker compose up

# 3) health + infer via Triton's KServe v2 endpoint
curl -s http://localhost:8000/v2/health/ready
curl -s -X POST http://localhost:8000/v2/models/llm_proxy/infer \
  -H 'Content-Type: application/json' -d '{
    "inputs": [
      {"name": "prompt",      "shape": [1], "datatype": "BYTES",  "data": ["What is an inference server?"]},
      {"name": "max_tokens",  "shape": [1], "datatype": "INT32",  "data": [64]},
      {"name": "temperature", "shape": [1], "datatype": "FP32",   "data": [0.7]}
    ]
  }'
```

## Compare against direct vLLM
Run the Phase 3 load tools against `localhost:8000` (Triton) and against
`:8000` (direct vLLM) and compare TTFT/ITL/throughput:
```bash
# NOTE: port conflict — run direct-vLLM vs Triton measurements separately, or
# map Triton to different host ports and adjust --url accordingly.
../../.venv-vllm/bin/python ../../loadtest/bench.py \
  --url http://localhost:8000/v2/models/llm_proxy/infer ...   # (HTTP/Kserve shape differs)
```
The proxy path is not OpenAI-shaped, so `bench.py` (OpenAI chat) targets vLLM
directly; for an apples-to-apples OpenAI comparison, front the proxy with an
OpenAI-compatible shim or use Triton's OpenAI frontend where available.

## Files
- `docker-compose.yml` — Triton service (host network, GPU reservation)
- `model_repository/llm_proxy/config.pbtxt` — model I/O config
- `model_repository/llm_proxy/1/model.py` — Python-backend proxy to vLLM
