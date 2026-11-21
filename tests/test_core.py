import pytest
import torch
import torch.nn as nn

from t2l import HyperLoRA, LoRAInjector, example_embedding, find_targets, load_peft_adapter, save_peft_adapter, task_embedding


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, 2 * d)
        self.v_proj = nn.Linear(d, d // 2)
        self.out = nn.Linear(2 * d + d // 2, d)

    def forward(self, x):
        return x + self.out(torch.cat([self.q_proj(x), self.v_proj(x)], -1))


class Toy(nn.Module):
    def __init__(self, d=16, n=3):
        super().__init__()
        self.layers = nn.ModuleList([Block(d) for _ in range(n)])
        self.vision = nn.Linear(d, d)  # never targeted

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return x


@pytest.fixture
def toy():
    torch.manual_seed(0)
    m = Toy()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def test_find_targets(toy):
    t = find_targets(toy, ["q_proj", "v_proj"], expected=6)
    assert [x.kind for x in t[:2]] == ["q_proj", "v_proj"]
    assert (t[0].in_features, t[0].out_features) == (16, 32)
    assert len(find_targets(toy, ["re:layers\\.0\\..*proj"])) == 2
    with pytest.raises(ValueError):
        find_targets(toy, ["q_proj"], expected=5)
    with pytest.raises(ValueError):
        find_targets(toy, ["nothing"])


def test_generator_shapes_and_zero_start(toy):
    t = find_targets(toy, ["q_proj", "v_proj"])
    g = HyperLoRA(t, emb_dim=10, rank=4, alpha=8, hidden=32, depth=2)
    w = g(torch.randn(10))
    a, b = w[t[0].name]
    assert a.shape == (4, 16) and b.shape == (32, 4) and g.scale == 2.0
    x = torch.randn(5, 16)
    with LoRAInjector(toy, t, g.scale) as inj:
        base = toy(x)
        inj.set(w)
        assert torch.allclose(toy(x), base)  # B starts at zero: no change before training
    with pytest.raises(ValueError):
        g(torch.randn(3, 10))


def test_injector_matches_merged_weights_and_detaches(toy):
    t = find_targets(toy, ["q_proj"])
    w = {s.name: (torch.randn(2, s.in_features), torch.randn(s.out_features, 2)) for s in t}
    x = torch.randn(4, 16)
    base = toy(x)
    with LoRAInjector(toy, t, scale=0.5) as inj:
        inj.set(w)
        adapted = toy(x)
    merged = Toy()
    merged.load_state_dict(toy.state_dict())
    mods = dict(merged.named_modules())
    for name, (a, b) in w.items():
        mods[name].weight.data += 0.5 * b @ a
    assert torch.allclose(adapted, merged(x), atol=1e-5)
    assert torch.allclose(toy(x), base)  # hooks removed on exit


def test_gradients_reach_generator(toy):
    t = find_targets(toy, ["q_proj", "v_proj"])
    g = HyperLoRA(t, emb_dim=8, rank=2, alpha=4, hidden=16, depth=1)
    opt = torch.optim.AdamW(g.parameters(), lr=1e-2)
    x, target = torch.randn(6, 16), torch.randn(6, 16)
    losses = []
    with LoRAInjector(toy, t, g.scale) as inj:
        for _ in range(30):
            inj.set(g(torch.ones(8)))
            loss = ((toy(x) - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert all(p.grad is None for p in toy.parameters())  # the base model stays frozen


def test_embeddings():
    h = torch.arange(12, dtype=torch.float32).view(4, 3)
    e = example_embedding(h, torch.tensor([1, 1, 1, 0]), torch.tensor([0, 0, 1, 1]))
    assert e.shape == (6,)
    assert torch.allclose(e[:3], h[:3].mean(0)) and torch.allclose(e[3:], h[2])
    assert torch.allclose(task_embedding(torch.stack([e, 3 * e])), 2 * e)
    with pytest.raises(ValueError):
        example_embedding(h, torch.ones(4), torch.zeros(4))


def test_peft_roundtrip(toy, tmp_path):
    t = find_targets(toy, ["q_proj", "v_proj"])
    w = {s.name: (torch.randn(4, s.in_features), torch.randn(s.out_features, 4)) for s in t}
    save_peft_adapter(w, t, tmp_path / "ad", rank=4, alpha=8, base_model="toy", dtype=torch.float32, extra={"k": 4})
    w2, cfg = load_peft_adapter(tmp_path / "ad", t)
    assert cfg["r"] == 4 and cfg["lora_alpha"] == 8 and cfg["target_modules"][0] == t[0].name and cfg["t2l"] == {"k": 4}
    for n in w:
        assert torch.equal(w[n][0], w2[n][0]) and torch.equal(w[n][1], w2[n][1])


def test_peft_library_loads_our_adapter(toy, tmp_path):
    peft = pytest.importorskip("peft")
    t = find_targets(toy, ["q_proj"])
    w = {s.name: (torch.randn(4, s.in_features), torch.randn(s.out_features, 4)) for s in t}
    save_peft_adapter(w, t, tmp_path / "ad", rank=4, alpha=8, dtype=torch.float32)
    x = torch.randn(3, 16)
    with LoRAInjector(toy, t, scale=2.0) as inj:
        inj.set(w)
        ours = toy(x)
    pm = peft.PeftModel.from_pretrained(toy, str(tmp_path / "ad"))
    assert torch.allclose(pm(x), ours, atol=1e-5)
