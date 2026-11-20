"""The hypernetwork: task embedding -> rank-r LoRA (A, B) for every target module.

Text-to-LoRA "M" style: one shared MLP is queried once per target module with
[task embedding, module-position embedding, module-type embedding]; a head per
module shape emits the flattened A and B.

Each module's adapter is `shared + task part`:
- shared: learned A0, B0 per module (a multi-task LoRA living inside the hypernetwork;
  B0 starts at zero like ordinary LoRA);
- task part: from the MLP; its B head starts at zero, so training starts from "no change".
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn

from t2l.targets import TargetSpec


class HyperLoRA(nn.Module):
    def __init__(self, targets: list[TargetSpec], emb_dim: int, rank: int = 8, alpha: float = 16,
                 hidden: int = 512, depth: int = 3, pos_dim: int = 64, kind_dim: int = 32, shared: bool = True):
        super().__init__()
        self.targets = list(targets)
        self.rank, self.alpha, self.emb_dim = rank, alpha, emb_dim
        self.hidden, self.depth, self.pos_dim, self.kind_dim = hidden, depth, pos_dim, kind_dim
        self.kinds = sorted({t.kind for t in targets})
        self.shapes = sorted({(t.in_features, t.out_features) for t in targets})
        self.norm = nn.LayerNorm(emb_dim)
        self.pos = nn.Embedding(len(targets), pos_dim)
        self.kind = nn.Embedding(len(self.kinds), kind_dim)
        layers, d = [], emb_dim + pos_dim + kind_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.mlp = nn.Sequential(*layers)
        self.head_a = nn.ModuleDict({self._key(i, o): nn.Linear(hidden, rank * i) for i, o in self.shapes})
        self.head_b = nn.ModuleDict({self._key(i, o): nn.Linear(hidden, o * rank) for i, o in self.shapes})
        for h in self.head_a.values():
            nn.init.normal_(h.weight, std=1.0 / math.sqrt(hidden * 4))
            nn.init.zeros_(h.bias)
        for h in self.head_b.values():
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self.shared = shared
        if shared:
            self.a0 = nn.ParameterList([nn.Parameter(torch.empty(rank, t.in_features)) for t in targets])
            self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(t.out_features, rank)) for t in targets])
            for p in self.a0:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.register_buffer("_kind_idx", torch.tensor([self.kinds.index(t.kind) for t in targets]), persistent=False)

    @staticmethod
    def _key(i: int, o: int) -> str:
        return f"{i}x{o}"

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def config(self) -> dict:
        """Everything needed to rebuild this module (see `t2l.checkpoint`)."""
        return {"rank": self.rank, "alpha": self.alpha, "emb_dim": self.emb_dim, "hidden": self.hidden,
                "depth": self.depth, "pos_dim": self.pos_dim, "kind_dim": self.kind_dim, "kinds": self.kinds,
                "targets": [dict(t.__dict__) for t in self.targets], "shared": self.shared}

    @classmethod
    def from_config(cls, cfg: dict) -> "HyperLoRA":
        targets = [TargetSpec(**t) for t in cfg["targets"]]
        return cls(targets, emb_dim=cfg["emb_dim"], rank=cfg["rank"], alpha=cfg["alpha"], hidden=cfg.get("hidden", 512),
                   depth=cfg.get("depth", 3), pos_dim=cfg.get("pos_dim", 64), kind_dim=cfg.get("kind_dim", 32),
                   shared=cfg.get("shared", True))

    def forward(self, task_emb: torch.Tensor) -> "OrderedDict[str, tuple[torch.Tensor, torch.Tensor]]":
        """task_emb: [emb_dim] for one task. Returns {module name: (A [r, in], B [out, r])}."""
        if task_emb.dim() != 1 or task_emb.shape[0] != self.emb_dim:
            raise ValueError(f"task embedding must have shape [{self.emb_dim}], got {tuple(task_emb.shape)}")
        n = len(self.targets)
        e = self.norm(task_emb.float()).expand(n, -1)
        idx = torch.arange(n, device=e.device)
        h = self.mlp(torch.cat([e, self.pos(idx), self.kind(self._kind_idx)], dim=-1))
        out: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        for j, t in enumerate(self.targets):
            key = self._key(t.in_features, t.out_features)
            a = self.head_a[key](h[j]).view(self.rank, t.in_features)
            b = self.head_b[key](h[j]).view(t.out_features, self.rank)
            if self.shared:
                a, b = a + self.a0[j], b + self.b0[j]
            out[t.name] = (a, b)
        return out
