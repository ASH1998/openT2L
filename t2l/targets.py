"""Find the linear layers an adapter attaches to, by module-name pattern."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import torch.nn as nn


@dataclass(frozen=True)
class TargetSpec:
    name: str          # full module name in the model, e.g. "model.layers.3.self_attn.q_proj"
    kind: str          # the pattern it matched, e.g. "self_attn.q_proj"; shared by all layers of that type
    in_features: int
    out_features: int


def find_targets(model: nn.Module, patterns: Iterable[str], exclude: Iterable[str] = (),
                 expected: int | None = None) -> list[TargetSpec]:
    """All `nn.Linear` modules whose name ends with one of `patterns` (or matches it as a regex
    when the pattern starts with "re:"), skipping names that contain any `exclude` substring.

    Order is the model's module order. Raises if `expected` is given and the count differs,
    or if nothing matches.
    """
    pats = list(patterns)
    excl = [e.lower() for e in exclude]
    out: list[TargetSpec] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or any(e in name.lower() for e in excl):
            continue
        for p in pats:
            hit = re.search(p[3:], name) if p.startswith("re:") else name.endswith(p)
            if hit:
                out.append(TargetSpec(name, p, mod.in_features, mod.out_features))
                break
    if not out:
        raise ValueError(f"no nn.Linear module matches {pats}")
    if expected is not None and len(out) != expected:
        raise ValueError(f"expected {expected} target modules, found {len(out)}: {[t.name for t in out[:6]]}...")
    return out
