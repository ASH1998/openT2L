"""Model-agnostic hypernetwork that emits LoRA adapters from a few examples.

Depends only on torch and safetensors. Works with any model whose adapted layers are `nn.Linear`:

    from t2l import find_targets, HyperLoRA, train_hypernet, emit_adapter, save_hypernet

    targets = find_targets(model, ["q_proj", "v_proj"])        # name patterns; shapes read from the model
    gen = HyperLoRA(targets, emb_dim=model_hidden_size, rank=8, alpha=16)
    train_hypernet(gen, model, batches, loss_fn, steps=6000)    # batches of Episode(support=[k, D], query=...)
    save_hypernet(gen, "hypernet/", base_model="org/model", embedding={"pooling": "answer", "layer": -1})
    emit_adapter(gen, example_embeddings, "my_adapter/")       # standard PEFT LoRA directory

How examples become embeddings is up to the caller; `example_embedding` is the default pooling
of one hidden layer.
"""

__version__ = "0.1.0"

from t2l.targets import TargetSpec, find_targets
from t2l.generator import HyperLoRA
from t2l.inject import LoRAInjector
from t2l.embed import example_embedding, task_embedding
from t2l.peft_io import save_peft_adapter, load_peft_adapter
from t2l.checkpoint import save_hypernet, load_hypernet
from t2l.emit import emit, emit_adapter
from t2l.train import Episode, train_hypernet

__all__ = ["TargetSpec", "find_targets", "HyperLoRA", "LoRAInjector", "example_embedding", "task_embedding",
           "save_peft_adapter", "load_peft_adapter", "save_hypernet", "load_hypernet", "emit", "emit_adapter",
           "Episode", "train_hypernet"]
