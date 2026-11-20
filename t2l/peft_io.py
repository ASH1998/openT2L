"""Save and load adapters in the standard PEFT LoRA layout.

The saved directory loads with `peft.PeftModel.from_pretrained(base_model, path)` and
converts with llama.cpp's `convert_lora_to_gguf.py`, like any PEFT LoRA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch

from t2l.targets import TargetSpec


def _key(name: str, ab: str) -> str:
    return f"base_model.model.{name}.lora_{ab}.weight"


def save_peft_adapter(weights: Mapping[str, tuple[torch.Tensor, torch.Tensor]], targets: list[TargetSpec],
                      path: str | Path, rank: int, alpha: float, base_model: str = "",
                      dtype: torch.dtype = torch.float16, extra: dict | None = None) -> Path:
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for t in targets:
        a, b = weights[t.name]
        if tuple(a.shape) != (rank, t.in_features) or tuple(b.shape) != (t.out_features, rank):
            raise ValueError(f"{t.name}: bad shapes A{tuple(a.shape)} B{tuple(b.shape)}")
        tensors[_key(t.name, "A")] = a.detach().to("cpu", dtype).contiguous()
        tensors[_key(t.name, "B")] = b.detach().to("cpu", dtype).contiguous()
    save_file(tensors, str(path / "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "task_type": None, "r": rank, "lora_alpha": alpha, "lora_dropout": 0.0,
           "bias": "none", "target_modules": [t.name for t in targets], "base_model_name_or_path": base_model,
           "fan_in_fan_out": False, "inference_mode": True, "use_rslora": False, "use_dora": False,
           **({"t2l": extra} if extra else {})}
    (path / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def load_peft_adapter(path: str | Path, targets: list[TargetSpec] | None = None,
                      device: str | torch.device = "cpu") -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
    """Returns ({module name: (A, B)}, adapter_config). Accepts adapters saved by PEFT itself
    (keys may lack the ".default" adapter name or include it)."""
    from safetensors.torch import load_file

    path = Path(path)
    cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    raw = load_file(str(path / "adapter_model.safetensors"), device=str(device))
    out: dict[str, list] = {}
    for k, v in raw.items():
        k = k.replace(".default", "")
        for ab, i in (("lora_A", 0), ("lora_B", 1)):
            tag = f".{ab}.weight"
            if k.endswith(tag):
                name = k[: -len(tag)].removeprefix("base_model.model.")
                out.setdefault(name, [None, None])[i] = v
    weights = {n: (a, b) for n, (a, b) in out.items()}
    if targets is not None:
        names = {t.name for t in targets}
        missing = names - set(weights)
        if missing:
            raise KeyError(f"adapter lacks modules: {sorted(missing)[:5]}")
        weights = {n: w for n, w in weights.items() if n in names}
    return weights, cfg
