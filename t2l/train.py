"""A model-agnostic training loop for the hypernetwork.

Each training step is a batch of episodes. An episode is a few support examples from one task
(already embedded, [k, D]) plus one query example from the same task. The hypernetwork turns
the support embeddings into an adapter, the adapter is injected into the frozen model, and
the loss on the query (computed by your `loss_fn`) trains the hypernetwork.

The MiniCPM-V document-extraction run (https://github.com/ASH1998/openBMB-spark-t2l) uses the same steps;
this version contains no model-specific code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn as nn

from t2l.embed import task_embedding
from t2l.generator import HyperLoRA
from t2l.inject import LoRAInjector


@dataclass
class Episode:
    support: torch.Tensor  # [k, D] example embeddings
    query: Any             # whatever `loss_fn` needs, e.g. a tokenised batch with labels


def cosine_schedule(opt: torch.optim.Optimizer, warmup: int, total: int) -> torch.optim.lr_scheduler.LambdaLR:
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def train_hypernet(gen: HyperLoRA, model: nn.Module, batches: Iterable[Sequence[Episode]],
                   loss_fn: Callable[[nn.Module, Any], torch.Tensor], *, steps: int, lr: float = 1e-4,
                   weight_decay: float = 0.0, warmup: int = 0, grad_clip: float | None = 1.0,
                   on_step: Callable[[int, float], None] | None = None) -> list[float]:
    """Train `gen` in place for `steps` batches. The model should be frozen (requires_grad False).
    Returns the per-step mean loss."""
    device = next(gen.parameters()).device
    opt = torch.optim.AdamW(gen.parameters(), lr=lr, weight_decay=weight_decay)
    sched = cosine_schedule(opt, warmup, steps)
    log: list[float] = []
    gen.train()
    with LoRAInjector(model, gen.targets, gen.scale) as inj:
        for step, batch in enumerate(batches, 1):
            if step > steps:
                break
            tot = 0.0
            for ep in batch:
                inj.set(gen(task_embedding(ep.support).to(device)))
                loss = loss_fn(model, ep.query) / len(batch)
                loss.backward()
                tot += loss.item()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(gen.parameters(), grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad()
            log.append(tot)
            if on_step:
                on_step(step, tot)
    gen.eval()
    return log
