from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from .k_reducer import KSemReduceConfig, k_semreduce
from .reducer import SemReduceResult, TokenNorm, _stack_results


@dataclass(frozen=True)
class AnswerAdaptiveKSemReduceConfig:
    """Configuration for Answer-Adaptive K-SemReduce.

    The number of output tokens is not fixed here. It is determined by the
    number of rows in the per-example semantic hypothesis matrix ``W_H``.
    """

    iterations: int = 3
    temperature: float = 0.1
    lambda_importance: float = 0.25
    gamma: float = 1.0
    eps: float = 1e-6
    sort_by_position: bool = True


class AnswerAdaptiveKSemReduce(nn.Module):
    """K-SemReduce with a per-question semantic hypothesis matrix."""

    def __init__(self, config: AnswerAdaptiveKSemReduceConfig, token_norm: TokenNorm = None) -> None:
        super().__init__()
        self.config = config
        if isinstance(token_norm, nn.Module):
            self.token_norm = token_norm
            self._callable_norm: Callable[[torch.Tensor], torch.Tensor] | None = None
        else:
            self.token_norm = None
            self._callable_norm = token_norm

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        hypothesis_embeddings: torch.Tensor,
        positions: torch.Tensor | None = None,
        cls_index: int = 0,
    ) -> SemReduceResult:
        if cls_index != 0:
            raise ValueError("AnswerAdaptiveKSemReduce currently expects the CLS token at index 0")
        if sequence_tokens.ndim == 2:
            return self._forward_sequence_single(sequence_tokens, hypothesis_embeddings, positions)
        if sequence_tokens.ndim != 3:
            raise ValueError(
                "Expected sequence_tokens with shape [N + 1, D] or [B, N + 1, D], "
                f"got {tuple(sequence_tokens.shape)}"
            )

        per_batch = []
        for batch_index, batch_tokens in enumerate(sequence_tokens):
            batch_positions = None if positions is None else positions[batch_index]
            batch_hypotheses = (
                hypothesis_embeddings
                if hypothesis_embeddings.ndim == 2
                else hypothesis_embeddings[batch_index]
            )
            per_batch.append(
                self._forward_sequence_single(batch_tokens, batch_hypotheses, batch_positions)
            )
        return _stack_results(per_batch, include_sequence=True)

    def _forward_sequence_single(
        self,
        sequence_tokens: torch.Tensor,
        hypothesis_embeddings: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> SemReduceResult:
        if sequence_tokens.ndim != 2:
            raise ValueError(f"Expected [N + 1, D] tokens, got {tuple(sequence_tokens.shape)}")
        if sequence_tokens.shape[0] < 2:
            raise ValueError("sequence_tokens must contain one CLS token and at least one patch token")

        cls_token = sequence_tokens[0]
        reduced = answer_adaptive_k_semreduce(
            patch_tokens=sequence_tokens[1:],
            hypothesis_embeddings=hypothesis_embeddings,
            cls_token=cls_token,
            config=self.config,
            token_norm=self._norm,
            positions=positions,
        )
        reduced.sequence = torch.cat([cls_token.unsqueeze(0), reduced.patch_tokens], dim=0)
        return reduced

    def _norm(self, values: torch.Tensor) -> torch.Tensor:
        if self.token_norm is not None:
            return self.token_norm(values)
        if self._callable_norm is not None:
            return self._callable_norm(values)
        return F.layer_norm(values.float(), values.shape[-1:])


def answer_adaptive_k_semreduce(
    patch_tokens: torch.Tensor,
    hypothesis_embeddings: torch.Tensor,
    cls_token: torch.Tensor | None = None,
    config: AnswerAdaptiveKSemReduceConfig | None = None,
    token_norm: TokenNorm = None,
    positions: torch.Tensor | None = None,
    **config_overrides: object,
) -> SemReduceResult:
    """Reduce tokens using a dynamic semantic hypothesis matrix.

    Args:
        patch_tokens: Tensor with shape ``[N, D]`` or ``[B, N, D]``.
        hypothesis_embeddings: Frozen semantic hypothesis matrix ``[K_Q, D]``
            or batched matrix ``[B, K_Q, D]``.
    """

    cfg = _resolve_answer_adaptive_config(config, config_overrides)
    _validate_hypothesis_embeddings(patch_tokens, hypothesis_embeddings)
    if patch_tokens.ndim == 2:
        return _answer_adaptive_single(
            patch_tokens,
            hypothesis_embeddings,
            cls_token,
            cfg,
            token_norm,
            positions,
        )
    if patch_tokens.ndim != 3:
        raise ValueError(
            f"Expected patch_tokens with shape [N, D] or [B, N, D], got {tuple(patch_tokens.shape)}"
        )

    per_batch = []
    for batch_index, batch_tokens in enumerate(patch_tokens):
        batch_cls = None if cls_token is None else cls_token[batch_index]
        batch_positions = None if positions is None else positions[batch_index]
        batch_hypotheses = (
            hypothesis_embeddings
            if hypothesis_embeddings.ndim == 2
            else hypothesis_embeddings[batch_index]
        )
        per_batch.append(
            _answer_adaptive_single(
                batch_tokens,
                batch_hypotheses,
                batch_cls,
                cfg,
                token_norm,
                batch_positions,
            )
        )
    return _stack_results(per_batch, include_sequence=False)


def _answer_adaptive_single(
    patch_tokens: torch.Tensor,
    hypothesis_embeddings: torch.Tensor,
    cls_token: torch.Tensor | None,
    cfg: AnswerAdaptiveKSemReduceConfig,
    token_norm: TokenNorm,
    positions: torch.Tensor | None,
) -> SemReduceResult:
    k_config = KSemReduceConfig(
        num_semantic_classes=int(hypothesis_embeddings.shape[0]),
        iterations=cfg.iterations,
        temperature=cfg.temperature,
        lambda_importance=cfg.lambda_importance,
        gamma=cfg.gamma,
        eps=cfg.eps,
        sort_by_position=cfg.sort_by_position,
    )
    return k_semreduce(
        patch_tokens=patch_tokens,
        classifier=hypothesis_embeddings,
        cls_token=cls_token,
        config=k_config,
        token_norm=token_norm,
        positions=positions,
    )


def _resolve_answer_adaptive_config(
    config: AnswerAdaptiveKSemReduceConfig | None,
    overrides: dict[str, object],
) -> AnswerAdaptiveKSemReduceConfig:
    if config is None:
        return AnswerAdaptiveKSemReduceConfig(**overrides)  # type: ignore[arg-type]
    if not overrides:
        return config
    values = {**config.__dict__, **overrides}
    return AnswerAdaptiveKSemReduceConfig(**values)


def _validate_hypothesis_embeddings(
    patch_tokens: torch.Tensor,
    hypothesis_embeddings: torch.Tensor,
) -> None:
    if hypothesis_embeddings.ndim not in {2, 3}:
        raise ValueError(
            "hypothesis_embeddings must have shape [K_Q, D] or [B, K_Q, D], "
            f"got {tuple(hypothesis_embeddings.shape)}"
        )
    if int(hypothesis_embeddings.shape[-2]) <= 0:
        raise ValueError("hypothesis_embeddings must contain at least one semantic hypothesis")
    if int(hypothesis_embeddings.shape[-1]) != int(patch_tokens.shape[-1]):
        raise ValueError(
            "hypothesis embedding dimension must match patch token dimension: "
            f"{hypothesis_embeddings.shape[-1]} vs {patch_tokens.shape[-1]}"
        )
    if patch_tokens.ndim == 3 and hypothesis_embeddings.ndim == 3:
        if int(patch_tokens.shape[0]) != int(hypothesis_embeddings.shape[0]):
            raise ValueError("batched patch_tokens and hypothesis_embeddings must share batch size")
