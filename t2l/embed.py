"""Default way to turn examples into a task embedding.

An example is one model input that ends with the correct answer (for a vision-language
model: image + prompt + gold output). Run the frozen model once, take one hidden layer,
and concatenate two means: over all tokens and over the answer tokens. A task embedding
is the mean of its examples' embeddings. Any other function with the same output shape
can be used instead.
"""

from __future__ import annotations

import torch


POOLINGS = ("both", "answer", "all")


def example_embedding(hidden: torch.Tensor, attention_mask: torch.Tensor, answer_mask: torch.Tensor,
                      pooling: str = "both") -> torch.Tensor:
    """hidden [T, d] (or [1, T, d]); masks [T] (or [1, T]) of 0/1. Returns float32 of shape
    [2d] for pooling="both" (all-token mean, answer-token mean), [d] for "answer" or "all"."""
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
    if hidden.dim() == 3:
        hidden, attention_mask, answer_mask = hidden[0], attention_mask.reshape(-1), answer_mask.reshape(-1)
    h = hidden.float()
    am = attention_mask.float().unsqueeze(-1)
    lm = (answer_mask.float() * attention_mask.float()).unsqueeze(-1)
    all_mean = (h * am).sum(0) / am.sum().clamp(min=1)
    if pooling == "all":
        return all_mean
    if lm.sum() == 0:
        raise ValueError("answer mask is empty")
    ans_mean = (h * lm).sum(0) / lm.sum()
    if pooling == "answer":
        return ans_mean
    return torch.cat([all_mean, ans_mean])


def task_embedding(example_embeddings: torch.Tensor) -> torch.Tensor:
    """[k, D] -> [D]: the mean over the k examples."""
    if example_embeddings.dim() != 2 or example_embeddings.shape[0] == 0:
        raise ValueError("need a non-empty [k, D] tensor")
    return example_embeddings.float().mean(0)
