"""One call from example embeddings to an adapter."""

from __future__ import annotations

from pathlib import Path

import torch

from t2l.embed import task_embedding
from t2l.generator import HyperLoRA
from t2l.peft_io import save_peft_adapter


@torch.no_grad()
def emit(gen: HyperLoRA, example_embeddings: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """example_embeddings: [k, D] (one row per example) or an already-pooled [D].
    Returns {module name: (A, B)} for use with `LoRAInjector` or `save_peft_adapter`."""
    e = example_embeddings if example_embeddings.dim() == 1 else task_embedding(example_embeddings)
    device = next(gen.parameters()).device
    return gen(e.to(device))


def emit_adapter(gen: HyperLoRA, example_embeddings: torch.Tensor, path: str | Path, *, base_model: str = "",
                 dtype: torch.dtype = torch.float16) -> Path:
    """Emit and save as a standard PEFT LoRA directory."""
    return save_peft_adapter(emit(gen, example_embeddings), gen.targets, path, rank=gen.rank, alpha=gen.alpha,
                             base_model=base_model, dtype=dtype)
