"""End-to-end openT2L on a text-only causal LM: train a small hypernetwork on toy string
tasks, emit an adapter for an unseen task from 4 examples, save it as PEFT and load it back.

    pip install "opent2l[hf]"
    python examples/quickstart_text.py --tiny                    # random tiny Llama + byte tokenizer, CPU, no download
    python examples/quickstart_text.py --model Qwen/Qwen3-0.6B --steps 2000 --device cuda

The toy tasks are deliberately simple; the point is the API. For the document-extraction recipe
used with MiniCPM-V, see https://github.com/ASH1998/openBMB-spark-t2l.
"""

from __future__ import annotations

import argparse
import random
import string

import torch

from t2l import (Episode, HyperLoRA, emit_adapter, example_embedding, find_targets, load_hypernet,
                       save_hypernet, task_embedding, train_hypernet)

TASKS = {
    "upper": str.upper,
    "reverse": lambda s: s[::-1],
    "first3": lambda s: s[:3],
    "double": lambda s: s + s,
    "vowels": lambda s: "".join(c for c in s if c in "aeiou") or "-",
    "sorted": lambda s: "".join(sorted(s)),
    "last3": lambda s: s[-3:],          # held out: never trained on
}
HELD_OUT = "last3"


def word(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 8)))


class CharTokenizer:
    """Byte-level stand-in so --tiny needs no download."""

    def __len__(self) -> int:
        return 256

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        ids = torch.tensor([list(text.encode("utf-8"))])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def load(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.tiny:
        from transformers import LlamaConfig, LlamaForCausalLM

        tok = CharTokenizer()
        cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2)
        model = LlamaForCausalLM(cfg)
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32)
    model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def encode(tok, x: str, y: str, device: str) -> dict:
    prompt, full = f"Input: {x}\nOutput:", f"Input: {x}\nOutput: {y}\n"
    p = tok(prompt, return_tensors="pt")["input_ids"]
    f = tok(full, return_tensors="pt")
    labels = f["input_ids"].clone()
    labels[:, : p.shape[1]] = -100
    return {"input_ids": f["input_ids"].to(device), "attention_mask": f["attention_mask"].to(device), "labels": labels.to(device)}


@torch.no_grad()
def embed(model, tok, x: str, y: str, device: str) -> torch.Tensor:
    b = encode(tok, x, y, device)
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
    return example_embedding(out.hidden_states[-1], b["attention_mask"], (b["labels"] != -100).long(), pooling="answer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="quickstart_out")
    args = ap.parse_args()
    rng = random.Random(0)
    model, tok = load(args)

    targets = find_targets(model, ["q_proj", "v_proj"])
    gen = HyperLoRA(targets, emb_dim=model.config.hidden_size, rank=8, alpha=16).to(args.device)
    print(f"{len(targets)} target modules, hypernetwork {sum(p.numel() for p in gen.parameters()) / 1e6:.1f}M params")

    train_tasks = [t for t in TASKS if t != HELD_OUT]

    def batches():
        while True:
            eps = []
            for t in rng.sample(train_tasks, 4):
                ws = [word(rng) for _ in range(5)]
                sup = torch.stack([embed(model, tok, w, TASKS[t](w), args.device) for w in ws[:4]])
                eps.append(Episode(support=sup, query=encode(tok, ws[4], TASKS[t](ws[4]), args.device)))
            yield eps

    def loss_fn(m, batch):
        return m(**batch).loss

    log = train_hypernet(gen, model, batches(), loss_fn, steps=args.steps, lr=1e-4, warmup=min(50, args.steps // 10),
                         on_step=lambda s, l: print(f"step {s} loss {l:.3f}") if s % 20 == 0 else None)
    print(f"loss first {log[0]:.3f} last {log[-1]:.3f}")

    save_hypernet(gen, f"{args.out}/hypernet", base_model=args.model if not args.tiny else "tiny-random",
                  embedding={"pooling": "answer", "layer": -1, "format": "Input: x\\nOutput: y"})
    gen, _ = load_hypernet(f"{args.out}/hypernet", device=args.device)

    # Emit an adapter for the held-out task from 4 examples, then evaluate on new inputs.
    shots = [word(rng) for _ in range(4)]
    sup = torch.stack([embed(model, tok, w, TASKS[HELD_OUT](w), args.device) for w in shots])
    path = emit_adapter(gen, sup, f"{args.out}/adapter_{HELD_OUT}")
    print(f"emitted adapter -> {path}")

    try:
        from peft import PeftModel
    except ImportError:
        return
    tests = [word(rng) for _ in range(20)]
    base_loss = sum(loss_fn(model, encode(tok, w, TASKS[HELD_OUT](w), args.device)).item() for w in tests) / len(tests)
    peft_model = PeftModel.from_pretrained(model, str(path)).eval()
    with torch.no_grad():
        ad_loss = sum(loss_fn(peft_model, encode(tok, w, TASKS[HELD_OUT](w), args.device)).item() for w in tests) / len(tests)
    print(f"held-out task '{HELD_OUT}': answer loss base {base_loss:.3f} -> emitted adapter (loaded with PEFT) {ad_loss:.3f}")
    _ = task_embedding  # re-exported for callers who pool examples themselves


if __name__ == "__main__":
    main()
