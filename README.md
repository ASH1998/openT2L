# openT2L

Open Text-to-LoRA (import name `t2l`).

[中文说明](README.zh-CN.md)

Show a frozen model a few examples of a new task, and get back a LoRA adapter for that task in one forward pass. No fine-tuning at use time.

A small hypernetwork reads embeddings of k examples (inputs together with their correct outputs) and emits LoRA weights for the frozen model. The adapter is saved in the standard PEFT format, so anything that loads a PEFT LoRA can use it: transformers, vLLM, or llama.cpp after its converter.

- Works with **any** PyTorch model whose adapted layers are `nn.Linear`: text LLMs, vision-language models, hybrid attention models.
- Core dependencies: `torch`, `safetensors`.
- Hypernetworks save to safetensors plus JSON, with no pickle. The JSON records how the examples were embedded.

## Install

```bash
pip install opent2l          # core
pip install "opent2l[hf]"    # + transformers, peft (for the example and for loading adapters)
```

Not on PyPI yet. Until then: `pip install "opent2l[hf] @ git+https://github.com/ASH1998/openT2L"`, or `pip install -e ".[hf]"` from a clone.

## Use

```python
import torch
from t2l import (find_targets, HyperLoRA, Episode, train_hypernet,
                 save_hypernet, load_hypernet, example_embedding, emit_adapter)

# 1. Pick the layers to adapt (name patterns, like PEFT's target_modules). Shapes are read from the model.
targets = find_targets(model, ["q_proj", "v_proj"])
gen = HyperLoRA(targets, emb_dim=model.config.hidden_size, rank=8, alpha=16)

# 2. Embed one example: run the frozen model on input + correct answer, pool a hidden layer.
def embed(batch):  # input_ids, attention_mask, labels (-100 outside the answer)
    with torch.no_grad():
        h = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                  output_hidden_states=True).hidden_states[-1]
    return example_embedding(h, batch["attention_mask"], batch["labels"] != -100, pooling="answer")

# 3. Train. Each episode = k support embeddings [k, D] + one query example from the same task.
train_hypernet(gen, model, batches, loss_fn=lambda m, q: m(**q).loss, steps=6000, lr=1e-4)
save_hypernet(gen, "hypernet/", base_model="org/model", embedding={"pooling": "answer", "layer": -1})

# 4. Use. A few examples of a new task -> a PEFT adapter directory.
gen, cfg = load_hypernet("hypernet/")
emit_adapter(gen, torch.stack([embed(b) for b in examples]), "adapter/")
```

```python
from peft import PeftModel
model = PeftModel.from_pretrained(model, "adapter/")
```

You can also apply emitted weights without PEFT, and swap them per batch, with `LoRAInjector`. Gradients flow back into the hypernetwork; this is how training works.

```python
from t2l import LoRAInjector, emit
with LoRAInjector(model, gen.targets, gen.scale) as inj:
    inj.set(emit(gen, support_embeddings))
    out = model(**batch)
```

[examples/quickstart_text.py](examples/quickstart_text.py) runs the whole loop on a causal LM. Add `--tiny` to run it offline on a random tiny Llama as a smoke test.

## API

| Name | What it does |
| --- | --- |
| `find_targets(model, patterns, exclude=(), expected=None)` | Matches `nn.Linear` modules by name suffix, or by regex with `re:`. Returns `TargetSpec`s. |
| `HyperLoRA(targets, emb_dim, rank=8, alpha=16, ...)` | The hypernetwork: task embedding → `{module: (A, B)}`. |
| `example_embedding(hidden, attention_mask, answer_mask, pooling)` | Default example embedding: `"answer"`, `"all"` or `"both"` (concatenated) means. |
| `task_embedding(embs)` | Mean over the k examples. |
| `train_hypernet(gen, model, batches, loss_fn, steps, ...)` | Training loop with AdamW, cosine schedule and gradient clipping. The model stays frozen. |
| `emit(gen, embs)` / `emit_adapter(gen, embs, path)` | Emit weights, or emit and save as PEFT. |
| `LoRAInjector(model, targets, scale)` | Applies weights through forward hooks, without copying the model. |
| `save_hypernet` / `load_hypernet` | Hypernetwork checkpoint as safetensors + JSON. |
| `save_peft_adapter` / `load_peft_adapter` | PEFT LoRA directory I/O. |

## How it works

- **Hypernetwork** (Text-to-LoRA "M" style). One MLP is queried once per target module with [task embedding, layer-position embedding, module-type embedding]. A head per weight shape outputs that module's A and B.
- **Starting point.** Each adapter is a learned shared part (in effect a multi-task LoRA inside the hypernetwork) plus a task-specific part. Both B terms start at zero, so training starts from "no change".
- **Task embedding.** By default, the mean of a hidden layer over each example's answer tokens, averaged over the k examples. Any function that returns a fixed-size vector works, as long as you use the same one in training and at use time.
- **Training.** Sample a task, embed k support examples, emit an adapter, inject it, and take the frozen model's loss on another example from the same task. Only the hypernetwork is trained.

## Results

openT2L was built for the MiniCPM Spark Program, for document extraction with MiniCPM-V-4.7-1B. On 78 held-out document tasks, adapters emitted from 4 example pages raised field-level F1 from 0.728 to 0.881 (+0.154, 95% CI +0.116 to +0.195), and 71 of the 78 tasks improved. That comparison is against the base model only. Protocol, data and caveats are in the [openBMB-spark-t2l](https://github.com/ASH1998/openBMB-spark-t2l) repository.

## Licence

Apache-2.0. Pretrained hypernetwork weights, when published, carry the licence of their training data. They are listed separately.
