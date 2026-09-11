# =============================================================================
# Triton Python-backend proxy model (Mode 2 front door).
#
# Forwards inference to a vLLM OpenAI server's /v1/chat/completions and returns
# the assistant text. This is a minimal, self-contained reference — enough to
# demonstrate the integration and be updated/extended on a real deployment.
#
# Environment (set on the triton container, see docker-compose.yml):
#   VLLM_URL    base URL of the vLLM OpenAI server (default http://127.0.0.1:8000)
#   VLLM_MODEL  model id vLLM serves
# =============================================================================
import json
import os

import requests
import triton_python_backend_utils as pb_utils

VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.logger.log_info(f"llm_proxy -> vLLM {VLLM_URL} model={VLLM_MODEL}")

    def execute(self, requests):
        responses = []
        for request in requests:
            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
            max_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "max_tokens")
            temperature_tensor = pb_utils.get_input_tensor_by_name(request, "temperature")

            prompt = prompt_tensor.as_numpy()[0].decode("utf-8")
            max_tokens = int(max_tokens_tensor.as_numpy()[0]) if max_tokens_tensor else 128
            temperature = (
                float(temperature_tensor.as_numpy()[0]) if temperature_tensor else 0.7
            )

            body = {
                "model": VLLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            try:
                r = requests.post(
                    f"{VLLM_URL}/v1/chat/completions", json=body, timeout=120
                )
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                text = f"[proxy error] {exc!r}"

            out = pb_utils.Tensor(
                "text", __import__("numpy").array([text.encode("utf-8")])
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    def finalize(self):
        pass
