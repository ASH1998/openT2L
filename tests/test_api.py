import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from t2l import (Episode, HyperLoRA, emit, emit_adapter, example_embedding, find_targets, load_hypernet,
                       load_peft_adapter, save_hypernet, train_hypernet)


class Toy(nn.Module):
    def __init__(self, d=16, n=2):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({"q_proj": nn.Linear(d, d), "v_proj": nn.Linear(d, d // 2)}) for _ in range(n)])
        self.head = nn.Linear(d // 2, d)

    def forward(self, x):
        for b in self.layers:
            x = x + self.head(b["v_proj"](b["q_proj"](x)))
        return x


@pytest.fixture
def toy():
    torch.manual_seed(0)
    m = Toy()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def test_checkpoint_roundtrip(toy, tmp_path):
    gen = HyperLoRA(find_targets(toy, ["q_proj", "v_proj"]), emb_dim=8, hidden=32, depth=2)
    with torch.no_grad():
        for p in gen.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    save_hypernet(gen, tmp_path / "h", base_model="toy", embedding={"pooling": "answer"})
    gen2, cfg = load_hypernet(tmp_path / "h")
    assert cfg["embedding"] == {"pooling": "answer"} and cfg["base_model"] == "toy" and cfg["hidden"] == 32
    e = torch.randn(8)
    for (a, b), (a2, b2) in zip(gen.eval()(e).values(), gen2(e).values()):
        assert torch.allclose(a, a2) and torch.allclose(b, b2)


def test_load_rejects_other_dirs(tmp_path):
    (tmp_path / "hypernet_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_hypernet(tmp_path)


def test_emit_adapter_roundtrip(toy, tmp_path):
    targets = find_targets(toy, ["q_proj", "v_proj"])
    gen = HyperLoRA(targets, emb_dim=8)
    embs = torch.randn(4, 8)
    w = emit(gen, embs)
    assert set(w) == {t.name for t in targets}
    assert torch.allclose(emit(gen, embs.mean(0))[targets[0].name][0], w[targets[0].name][0])
    w2, cfg = load_peft_adapter(emit_adapter(gen, embs, tmp_path / "a", base_model="toy"), targets)
    assert cfg["r"] == 8 and cfg["base_model_name_or_path"] == "toy"
    assert torch.allclose(w2[targets[0].name][0].float(), w[targets[0].name][0].half().float())


def test_train_hypernet_learns_task_dependent_shift(toy):
    """Two tasks need opposite output shifts; the loss only falls if the adapter depends on the support set."""
    torch.manual_seed(1)
    gen = HyperLoRA(find_targets(toy, ["q_proj", "v_proj"]), emb_dim=4, hidden=64, depth=2, shared=False)
    x = torch.randn(32, 16)
    base = toy(x)
    task = {0: (torch.tensor([1.0, 0, 0, 0]), 1.0), 1: (torch.tensor([0, 1.0, 0, 0]), -1.0)}

    def batches():
        while True:
            yield [Episode(support=task[t][0].expand(2, 4), query=t) for t in (0, 1)]

    def loss_fn(model, t):
        return ((model(x) - (base + task[t][1])) ** 2).mean()

    log = train_hypernet(gen, toy, batches(), loss_fn, steps=300, lr=3e-3, warmup=10)
    assert len(log) == 300
    # Any adapter that ignores the support set scores >= 1.0 averaged over the two symmetric tasks.
    assert log[0] == pytest.approx(1.0, abs=0.05)
    assert sum(log[-20:]) / 20 < 0.6


def test_pooling_variants():
    h = torch.arange(12, dtype=torch.float32).view(3, 4)
    am, lm = torch.ones(3), torch.tensor([0, 0, 1])
    assert example_embedding(h, am, lm, "both").shape == (8,)
    assert torch.equal(example_embedding(h, am, lm, "answer"), h[2])
    assert torch.equal(example_embedding(h, am, torch.zeros(3), "all"), h.mean(0))
    with pytest.raises(ValueError):
        example_embedding(h, am, lm, "max")


def test_core_imports_without_heavy_deps():
    code = "import sys, t2l; bad = [m for m in ('modal', 'transformers', 'peft', 'datasets') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
