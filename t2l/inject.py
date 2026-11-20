"""Apply emitted LoRA weights to a frozen model without copying it.

Forward hooks on the target `nn.Linear` modules add `scale * B(A(x))` to their output.
Weights can be swapped per forward pass (one task at a time), and gradients flow back
into whatever produced them (the hypernetwork).
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from t2l.targets import TargetSpec


class LoRAInjector:
    def __init__(self, model: nn.Module, targets: list[TargetSpec], scale: float):
        self.model, self.targets, self.scale = model, list(targets), float(scale)
        self._mods = dict(model.named_modules())
        missing = [t.name for t in self.targets if t.name not in self._mods]
        if missing:
            raise KeyError(f"target modules not in model: {missing[:5]}")
        self._weights: Mapping[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._handles: list = []

    def _hook(self, name: str):
        def hook(module, inputs, output):
            w = self._weights.get(name)
            if w is None:
                return output
            a, b = w
            x = inputs[0]
            delta = F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype))
            return output + self.scale * delta
        return hook

    def attach(self) -> "LoRAInjector":
        if not self._handles:
            self._handles = [self._mods[t.name].register_forward_hook(self._hook(t.name)) for t in self.targets]
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._weights = {}

    def set(self, weights: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Use these (A, B) per module name from now on; modules not in `weights` get no adapter."""
        unknown = set(weights) - {t.name for t in self.targets}
        if unknown:
            raise KeyError(f"weights for unknown modules: {sorted(unknown)[:5]}")
        self._weights = weights

    def clear(self) -> None:
        self._weights = {}

    def __enter__(self) -> "LoRAInjector":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()
