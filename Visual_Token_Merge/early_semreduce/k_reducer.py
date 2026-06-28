from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from .reducer import (
    SemReduceResult,
    TokenNorm,
    _classifier_weight,
    _make_norm,
    _position_order,
    _prepare_positions,
    _remap_assignments,
    _semantic_response,
    _stack_results,
)


@dataclass(frozen=True)
class KSemReduceConfig:
    """Configuration for K-SemReduce.

    ``num_semantic_classes`` is the single core control variable. It is used as
    the Top-K candidate class count, semantic response dimension, cluster count,
    and output prototype-token count.
    """

    num_semantic_classes: int = 64
    iterations: int = 3
    temperature: float = 0.1
    lambda_importance: float = 0.25
    gamma: float = 1.0
    eps: float = 1e-6
    sort_by_position: bool = True


class KSemReduce(nn.Module):
    """Training-free class-prototype guided visual token reduction."""

    def __init__(self, config: KSemReduceConfig, token_norm: TokenNorm = None) -> None:
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
        classifier: torch.Tensor | nn.Module,
        positions: torch.Tensor | None = None,
        cls_index: int = 0,
    ) -> SemReduceResult:
        if cls_index != 0:
            raise ValueError("KSemReduce currently expects the CLS token at index 0")
        if sequence_tokens.ndim == 2:
            return self._forward_sequence_single(sequence_tokens, classifier, positions)
        if sequence_tokens.ndim != 3:
            raise ValueError(
                "Expected sequence_tokens with shape [N + 1, D] or [B, N + 1, D], "
                f"got {tuple(sequence_tokens.shape)}"
            )

        per_batch = []
        for batch_index, batch_tokens in enumerate(sequence_tokens):
            batch_positions = None if positions is None else positions[batch_index]
            per_batch.append(self._forward_sequence_single(batch_tokens, classifier, batch_positions))
        return _stack_results(per_batch, include_sequence=True)

    def _forward_sequence_single(
        self,
        sequence_tokens: torch.Tensor,
        classifier: torch.Tensor | nn.Module,
        positions: torch.Tensor | None,
    ) -> SemReduceResult:
        if sequence_tokens.ndim != 2:
            raise ValueError(f"Expected [N + 1, D] tokens, got {tuple(sequence_tokens.shape)}")
        if sequence_tokens.shape[0] < 2:
            raise ValueError("sequence_tokens must contain one CLS token and at least one patch token")

        cls_token = sequence_tokens[0]
        patch_tokens = sequence_tokens[1:]
        reduced = k_semreduce(
            patch_tokens=patch_tokens,
            classifier=classifier,
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


def k_semreduce(
    patch_tokens: torch.Tensor,
    classifier: torch.Tensor | nn.Module,
    cls_token: torch.Tensor | None = None,
    config: KSemReduceConfig | None = None,
    token_norm: TokenNorm = None,
    positions: torch.Tensor | None = None,
    **config_overrides: object,
) -> SemReduceResult:
    """Reduce patch tokens with K-SemReduce."""

    cfg = _resolve_k_config(config, config_overrides)
    if patch_tokens.ndim == 2:
        return _k_semreduce_single(patch_tokens, classifier, cls_token, cfg, token_norm, positions)
    if patch_tokens.ndim != 3:
        raise ValueError(
            f"Expected patch_tokens with shape [N, D] or [B, N, D], got {tuple(patch_tokens.shape)}"
        )

    per_batch = []
    for batch_index, batch_tokens in enumerate(patch_tokens):
        batch_cls = None if cls_token is None else cls_token[batch_index]
        batch_positions = None if positions is None else positions[batch_index]
        per_batch.append(
            _k_semreduce_single(
                batch_tokens,
                classifier,
                batch_cls,
                cfg,
                token_norm,
                batch_positions,
            )
        )
    return _stack_results(per_batch, include_sequence=False)


def _k_semreduce_single(
    patch_tokens: torch.Tensor,
    classifier: torch.Tensor | nn.Module,
    cls_token: torch.Tensor | None,
    cfg: KSemReduceConfig,
    token_norm: TokenNorm,
    positions: torch.Tensor | None,
) -> SemReduceResult:
    _validate_k_config(cfg)
    if patch_tokens.ndim != 2:
        raise ValueError(f"Expected [N, D] patch tokens, got {tuple(patch_tokens.shape)}")
    if patch_tokens.shape[0] == 0:
        raise ValueError("cannot reduce an empty patch-token sequence")

    num_patches = int(patch_tokens.shape[0])
    norm_fn = _make_norm(token_norm)
    weight = _classifier_weight(classifier, patch_tokens.device)
    target = min(int(cfg.num_semantic_classes), num_patches, int(weight.shape[0]))

    q_tokens, p_hat, importance, selected_classes = _semantic_response(
        patch_tokens=patch_tokens,
        classifier_weight=weight,
        cls_token=cls_token,
        token_norm=norm_fn,
        candidate_classes=target,
        eps=cfg.eps,
    )
    seed_indices = _select_class_guided_seeds(p_hat)
    centers = q_tokens[seed_indices].clone()
    centers, assignments = _cluster_with_repair(
        q_tokens=q_tokens,
        importance=importance,
        centers=centers,
        iterations=cfg.iterations,
        gamma=cfg.gamma,
        eps=cfg.eps,
    )

    return _aggregate_k_prototypes(
        patch_tokens=patch_tokens,
        q_tokens=q_tokens,
        importance=importance,
        centers=centers,
        assignments=assignments,
        seed_indices=seed_indices,
        selected_classes=selected_classes,
        positions=positions,
        temperature=cfg.temperature,
        lambda_importance=cfg.lambda_importance,
        sort_by_position=cfg.sort_by_position,
    )


def _select_class_guided_seeds(p_hat: torch.Tensor) -> torch.Tensor:
    num_patches, num_classes = p_hat.shape
    if num_classes > num_patches:
        raise ValueError("K-SemReduce requires K <= number of patch tokens after clamping")

    selected: list[int] = []
    unavailable = torch.zeros(num_patches, dtype=torch.bool, device=p_hat.device)
    for class_index in range(num_classes):
        scores = p_hat[:, class_index].clone()
        scores[unavailable] = -torch.inf
        patch_index = int(torch.argmax(scores).item())
        selected.append(patch_index)
        unavailable[patch_index] = True
    return torch.tensor(selected, dtype=torch.long, device=p_hat.device)


def _cluster_with_repair(
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    centers: torch.Tensor,
    iterations: int,
    gamma: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assignments = torch.zeros(q_tokens.shape[0], dtype=torch.long, device=q_tokens.device)
    for _ in range(max(0, int(iterations))):
        assignments = _assign_to_centers(q_tokens, centers)
        assignments = _repair_empty_clusters(q_tokens, centers, assignments)
        centers = _update_centers(q_tokens, importance, centers, assignments, gamma, eps)

    assignments = _assign_to_centers(q_tokens, centers)
    assignments = _repair_empty_clusters(q_tokens, centers, assignments)
    centers = _update_centers(q_tokens, importance, centers, assignments, gamma, eps)
    return centers, assignments


def _assign_to_centers(q_tokens: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (q_tokens @ centers.T).argmax(dim=-1)


def _repair_empty_clusters(
    q_tokens: torch.Tensor,
    centers: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    num_centers = int(centers.shape[0])
    counts = torch.bincount(assignments, minlength=num_centers)
    empty_centers = torch.nonzero(counts == 0, as_tuple=False).flatten()
    if empty_centers.numel() == 0:
        return assignments

    repaired = assignments.clone()
    for empty_center in empty_centers.tolist():
        donor_clusters = torch.nonzero(counts > 1, as_tuple=False).flatten()
        if donor_clusters.numel() == 0:
            raise RuntimeError("cannot repair empty K-SemReduce cluster without a non-singleton donor")

        donor = _select_donor_cluster(q_tokens, centers, repaired, donor_clusters)
        donor_members = torch.nonzero(repaired == donor, as_tuple=False).flatten()
        donor_similarities = q_tokens[donor_members] @ centers[donor]
        moved_patch = donor_members[torch.argmin(donor_similarities)]

        counts[donor] -= 1
        repaired[moved_patch] = int(empty_center)
        counts[empty_center] += 1
    return repaired


def _select_donor_cluster(
    q_tokens: torch.Tensor,
    centers: torch.Tensor,
    assignments: torch.Tensor,
    donor_clusters: torch.Tensor,
) -> int:
    best_cluster = int(donor_clusters[0].item())
    best_dispersion = -float("inf")
    for cluster_index in donor_clusters.tolist():
        members = torch.nonzero(assignments == int(cluster_index), as_tuple=False).flatten()
        similarities = q_tokens[members] @ centers[int(cluster_index)]
        dispersion = float((1.0 - similarities).mean().item())
        if dispersion > best_dispersion:
            best_dispersion = dispersion
            best_cluster = int(cluster_index)
    return best_cluster


def _update_centers(
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    centers: torch.Tensor,
    assignments: torch.Tensor,
    gamma: float,
    eps: float,
) -> torch.Tensor:
    updated = []
    for center_index in range(int(centers.shape[0])):
        member_mask = assignments == center_index
        if not bool(member_mask.any()):
            raise RuntimeError("K-SemReduce cluster repair left an empty cluster")
        weights = torch.exp(float(gamma) * importance[member_mask]).unsqueeze(-1)
        weighted = (weights * q_tokens[member_mask]).sum(dim=0, keepdim=True)
        updated.append(F.normalize(weighted, p=2, dim=-1, eps=eps).squeeze(0))
    return torch.stack(updated, dim=0)


def _aggregate_k_prototypes(
    patch_tokens: torch.Tensor,
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    centers: torch.Tensor,
    assignments: torch.Tensor,
    seed_indices: torch.Tensor,
    selected_classes: torch.Tensor,
    positions: torch.Tensor | None,
    temperature: float,
    lambda_importance: float,
    sort_by_position: bool,
) -> SemReduceResult:
    prototypes = []
    masses = []
    soft_positions = []
    prepared_positions = _prepare_positions(positions, int(patch_tokens.shape[0]), patch_tokens.device)

    for center_index in range(int(centers.shape[0])):
        member_indices = torch.nonzero(assignments == center_index, as_tuple=False).flatten()
        if member_indices.numel() == 0:
            raise RuntimeError("K-SemReduce cannot aggregate an empty cluster")

        semantic_scores = q_tokens[member_indices] @ centers[center_index]
        scores = semantic_scores + float(lambda_importance) * importance[member_indices]
        weights = torch.softmax(scores / float(temperature), dim=0).to(dtype=patch_tokens.dtype)

        prototypes.append((patch_tokens[member_indices] * weights.unsqueeze(-1)).sum(dim=0))
        masses.append(torch.tensor(member_indices.numel(), dtype=torch.long, device=patch_tokens.device))
        if prepared_positions is not None:
            pos_weights = weights.to(dtype=prepared_positions.dtype)
            soft_positions.append(
                (prepared_positions[member_indices] * pos_weights.unsqueeze(-1)).sum(dim=0)
            )

    patch_result = torch.stack(prototypes, dim=0).to(dtype=patch_tokens.dtype)
    masses_result = torch.stack(masses, dim=0)
    centers_result = centers
    positions_result = torch.stack(soft_positions, dim=0) if soft_positions else None
    prototype_order = seed_indices

    if sort_by_position and positions_result is not None:
        order = _position_order(positions_result)
        patch_result = patch_result[order]
        masses_result = masses_result[order]
        centers_result = centers_result[order]
        positions_result = positions_result[order]
        prototype_order = prototype_order[order]
        assignments = _remap_assignments(assignments, order)

    return SemReduceResult(
        sequence=None,
        patch_tokens=patch_result,
        assignments=assignments,
        centers=centers_result,
        selected_classes=selected_classes,
        anchors=torch.empty(0, dtype=torch.long, device=patch_tokens.device),
        masses=masses_result,
        soft_positions=positions_result,
        prototype_order=prototype_order,
    )


def _resolve_k_config(
    config: KSemReduceConfig | None,
    overrides: dict[str, object],
) -> KSemReduceConfig:
    if config is None:
        return KSemReduceConfig(**overrides)  # type: ignore[arg-type]
    if not overrides:
        return config
    values = {**config.__dict__, **overrides}
    return KSemReduceConfig(**values)


def _validate_k_config(cfg: KSemReduceConfig) -> None:
    if cfg.num_semantic_classes <= 0:
        raise ValueError("num_semantic_classes must be positive")
    if cfg.iterations < 0:
        raise ValueError("iterations must be non-negative")
    if cfg.temperature <= 0:
        raise ValueError("temperature must be positive")
    if cfg.eps <= 0:
        raise ValueError("eps must be positive")
