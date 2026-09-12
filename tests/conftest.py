"""Test fixtures: make repo modules importable without packaging.

`loadtest/bench.py` and `v1-baseline/main-torch.py` are standalone scripts (not a
package), so we add their directories to sys.path. For the baseline we stub the
heavy GPU/framework imports (torch, transformers, uvicorn) so the module can be
imported — and its Pydantic request/response models exercised — without a GPU or
model download.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def bench():
    """Import loadtest/bench.py (stdlib + aiohttp only)."""
    return _load_module("bench_under_test", REPO / "loadtest" / "bench.py")


@pytest.fixture(scope="session")
def baseline():
    """Import v1-baseline/main-torch.py with torch/transformers/uvicorn stubbed."""
    # Stub heavy imports so module import does not require a GPU or model.
    for modname in ("torch", "transformers", "uvicorn"):
        if modname in sys.modules:
            continue
        stub = types.ModuleType(modname)
        if modname == "torch":
            stub.device = lambda *a, **k: types.SimpleNamespace(type="cpu")
            stub.cuda = types.SimpleNamespace(
                is_available=lambda: False, get_device_name=lambda *a: "stub"
            )
            stub.backends = types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: False)
            )
            stub.float16 = "float16"
            stub.float32 = "float32"
            stub.no_grad = lambda: types.SimpleNamespace(
                __enter__=lambda s: s, __exit__=lambda *a: False
            )
        if modname == "transformers":
            stub.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})
            stub.AutoTokenizer = type("AutoTokenizer", (), {})
        if modname == "uvicorn":
            stub.run = lambda *a, **k: None
        sys.modules[modname] = stub

    # main-torch.py has a hyphen in the filename; import it via importlib.
    return _load_module("main_torch_under_test", REPO / "v1-baseline" / "main-torch.py")
