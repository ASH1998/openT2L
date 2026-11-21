"""Save and load a trained hypernetwork as plain safetensors + JSON (no pickle).

A hypernetwork directory holds:
- `hypernet.safetensors`: the weights;
- `hypernet_config.json`: the architecture (`HyperLoRA.config()`), plus `embedding` (how
  example embeddings were made, e.g. {"pooling": "answer", "layer": -1}) and `base_model`.

The embedding recipe matters: emitted adapters are only meaningful when examples are
embedded the same way, with the same frozen base model, as during training.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from t2l.generator import HyperLoRA

WEIGHTS = "hypernet.safetensors"
CONFIG = "hypernet_config.json"


def save_hypernet(gen: HyperLoRA, path: str | Path, *, base_model: str = "", embedding: dict | None = None,
                  extra: dict | None = None) -> Path:
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in gen.state_dict().items()}, str(path / WEIGHTS))
    cfg = {"format": "t2l-hypernet", "version": 1, **gen.config(), "base_model": base_model,
           "embedding": embedding or {}, **({"extra": extra} if extra else {})}
    (path / CONFIG).write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_hypernet(path: str | Path, device: str | torch.device = "cpu") -> tuple[HyperLoRA, dict]:
    """Returns (hypernetwork in eval mode, its config dict)."""
    from safetensors.torch import load_file

    path = Path(path)
    cfg = json.loads((path / CONFIG).read_text(encoding="utf-8"))
    if cfg.get("format") != "t2l-hypernet":
        raise ValueError(f"{path} is not a t2l hypernetwork directory")
    gen = HyperLoRA.from_config(cfg)
    gen.load_state_dict(load_file(str(path / WEIGHTS)))
    return gen.to(device).eval(), cfg
