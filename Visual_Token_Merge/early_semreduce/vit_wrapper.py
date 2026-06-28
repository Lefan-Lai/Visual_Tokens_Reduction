from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .reducer import EarlySemReduce, SemReduceResult


@dataclass
class TimmSemReduceOutput:
    logits: torch.Tensor
    features: torch.Tensor
    reduction: SemReduceResult


def forward_timm_vit_with_semreduce(
    model: nn.Module,
    images: torch.Tensor,
    reducer: EarlySemReduce,
    reduction_layer: int,
    classifier: torch.Tensor | nn.Module | None = None,
    return_info: bool = False,
) -> torch.Tensor | TimmSemReduceOutput:
    """Run a timm-style ViT with Early-SemReduce inserted after a block.

    This helper targets common timm VisionTransformer modules. It uses
    ``patch_embed``, ``blocks``, ``norm`` and ``head`` when present, and falls
    back to ``forward_head`` if the model provides it.
    """

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("model must expose a blocks attribute")
    if not 0 <= reduction_layer <= len(blocks):
        raise ValueError(f"reduction_layer must be in [0, {len(blocks)}]")

    x = _patch_embed(model, images)
    x = _add_position_tokens(model, x)

    for block in blocks[:reduction_layer]:
        x = block(x)

    classifier = classifier if classifier is not None else _default_classifier(model)
    reduction = reducer(x, classifier=classifier)
    if reduction.sequence is None:
        raise RuntimeError("EarlySemReduce did not return a reduced sequence")
    x = reduction.sequence

    for block in blocks[reduction_layer:]:
        x = block(x)

    logits, features = _forward_head(model, x)
    if return_info:
        return TimmSemReduceOutput(logits=logits, features=features, reduction=reduction)
    return logits


def _patch_embed(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    if not hasattr(model, "patch_embed"):
        raise ValueError("model must expose patch_embed")
    x = model.patch_embed(images)
    if x.ndim == 4:
        x = x.flatten(2).transpose(1, 2)
    return x


def _add_position_tokens(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "_pos_embed"):
        x = model._pos_embed(x)
    else:
        if hasattr(model, "cls_token") and model.cls_token is not None:
            cls_token = model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        if hasattr(model, "pos_embed") and model.pos_embed is not None:
            x = x + model.pos_embed[:, : x.shape[1]]

    if hasattr(model, "pos_drop"):
        x = model.pos_drop(x)
    if hasattr(model, "patch_drop"):
        x = model.patch_drop(x)
    if hasattr(model, "norm_pre"):
        x = model.norm_pre(x)
    return x


def _default_classifier(model: nn.Module) -> torch.Tensor | nn.Module:
    head = getattr(model, "head", None)
    if head is None or not hasattr(head, "weight"):
        raise ValueError("classifier was not provided and model.head.weight is unavailable")
    return head


def _forward_head(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "forward_head"):
        logits = model.forward_head(x)
        return logits, x

    features = model.norm(x) if hasattr(model, "norm") else x
    pooled = features[:, 0]
    if hasattr(model, "head"):
        return model.head(pooled), features
    return pooled, features
