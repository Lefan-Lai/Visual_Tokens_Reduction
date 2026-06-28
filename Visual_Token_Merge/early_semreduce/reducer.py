from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


TokenNorm = Callable[[torch.Tensor], torch.Tensor] | nn.Module | None


@dataclass(frozen=True)
class SemReduceConfig:
    """Configuration for Early-SemReduce."""

    num_prototypes: int
    candidate_classes: int | None = 64
    num_anchors: int = 8
    iterations: int = 5
    temperature: float = 0.07
    lambda_importance: float = 0.25
    lambda_diversity: float = 1.0
    gamma: float = 1.0
    eps: float = 1e-6
    sort_by_position: bool = True


@dataclass
class SemReduceResult:
    """Outputs and bookkeeping from Early-SemReduce.

    Attributes:
        sequence: Reduced sequence with CLS token when sequence input is used.
        patch_tokens: Reduced patch/prototype tokens.
        assignments: Prototype index for every original patch token.
        centers: Final semantic cluster centers for non-anchor clusters.
        selected_classes: Candidate class indices used for semantic responses.
        anchors: Original patch indices protected as singleton anchors.
        masses: Number of original patches represented by each prototype.
        soft_positions: Optional soft 2D positions for the prototypes.
        prototype_order: Order used when prototypes are sorted by soft position.
    """

    sequence: torch.Tensor | None
    patch_tokens: torch.Tensor
    assignments: torch.Tensor
    centers: torch.Tensor
    selected_classes: torch.Tensor
    anchors: torch.Tensor
    masses: torch.Tensor
    soft_positions: torch.Tensor | None
    prototype_order: torch.Tensor


class EarlySemReduce(nn.Module):
    """Training-free semantic response guided token reduction.

    The module expects an intermediate ViT sequence of shape ``[B, N + 1, D]``
    or ``[N + 1, D]`` where the first token is CLS. Patch tokens are clustered
    in classifier-response space, then merged in the original visual-token
    space.
    """

    def __init__(self, config: SemReduceConfig, token_norm: TokenNorm = None) -> None:
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
        """Reduce a CLS-plus-patch sequence.

        Args:
            sequence_tokens: Tensor with shape ``[N + 1, D]`` or
                ``[B, N + 1, D]``.
            classifier: Frozen classifier head weight ``[K, D]`` or an
                ``nn.Linear``/module exposing ``.weight``.
            positions: Optional patch positions with shape ``[N, 2]`` or
                ``[B, N, 2]``.
            cls_index: Index of the CLS token in ``sequence_tokens``.
        """

        if cls_index != 0:
            raise ValueError("EarlySemReduce currently expects the CLS token at index 0")
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
        reduced = early_semreduce(
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


def early_semreduce(
    patch_tokens: torch.Tensor,
    classifier: torch.Tensor | nn.Module,
    cls_token: torch.Tensor | None = None,
    config: SemReduceConfig | None = None,
    token_norm: TokenNorm = None,
    positions: torch.Tensor | None = None,
    **config_overrides: object,
) -> SemReduceResult:
    """Reduce patch tokens to semantic prototype tokens.

    Args:
        patch_tokens: Tensor with shape ``[N, D]`` or ``[B, N, D]``.
        classifier: Frozen classifier head weight ``[K, D]`` or module with
            ``.weight``.
        cls_token: Optional CLS tensor ``[D]`` or ``[B, D]`` used to select
            candidate classes. If omitted, the mean patch token is used.
        config: Optional ``SemReduceConfig``. Keyword overrides can also be
            provided, for example ``num_prototypes=64``.
        token_norm: Frozen model norm, custom callable, or ``None`` for
            parameter-free layer norm.
        positions: Optional patch positions with shape ``[N, 2]`` or
            ``[B, N, 2]``.
    """

    cfg = _resolve_config(config, config_overrides)
    if patch_tokens.ndim == 2:
        return _early_semreduce_single(patch_tokens, classifier, cls_token, cfg, token_norm, positions)
    if patch_tokens.ndim != 3:
        raise ValueError(
            f"Expected patch_tokens with shape [N, D] or [B, N, D], got {tuple(patch_tokens.shape)}"
        )

    per_batch = []
    for batch_index, batch_tokens in enumerate(patch_tokens):
        batch_cls = None if cls_token is None else cls_token[batch_index]
        batch_positions = None if positions is None else positions[batch_index]
        per_batch.append(
            _early_semreduce_single(
                batch_tokens,
                classifier,
                batch_cls,
                cfg,
                token_norm,
                batch_positions,
            )
        )
    return _stack_results(per_batch, include_sequence=False)


def _early_semreduce_single(
    patch_tokens: torch.Tensor,
    classifier: torch.Tensor | nn.Module,
    cls_token: torch.Tensor | None,
    cfg: SemReduceConfig,
    token_norm: TokenNorm,
    positions: torch.Tensor | None,
) -> SemReduceResult:
    _validate_config(cfg)
    if patch_tokens.ndim != 2:
        raise ValueError(f"Expected [N, D] patch tokens, got {tuple(patch_tokens.shape)}")
    if patch_tokens.shape[0] == 0:
        raise ValueError("cannot reduce an empty patch-token sequence")

    num_patches = int(patch_tokens.shape[0])
    target = min(int(cfg.num_prototypes), num_patches)
    norm_fn = _make_norm(token_norm)
    weight = _classifier_weight(classifier, patch_tokens.device)
    q_tokens, p_hat, importance, selected_classes = _semantic_response(
        patch_tokens=patch_tokens,
        classifier_weight=weight,
        cls_token=cls_token,
        token_norm=norm_fn,
        candidate_classes=cfg.candidate_classes,
        eps=cfg.eps,
    )

    if target == num_patches:
        order = torch.arange(num_patches, device=patch_tokens.device)
        masses = torch.ones(num_patches, dtype=torch.long, device=patch_tokens.device)
        result_positions = _prepare_positions(positions, num_patches, patch_tokens.device)
        return SemReduceResult(
            sequence=None,
            patch_tokens=patch_tokens,
            assignments=order,
            centers=q_tokens,
            selected_classes=selected_classes,
            anchors=torch.empty(0, dtype=torch.long, device=patch_tokens.device),
            masses=masses,
            soft_positions=result_positions,
            prototype_order=order,
        )

    num_anchors = min(max(int(cfg.num_anchors), 0), target)
    anchor_indices = _topk_indices(importance, num_anchors)
    non_anchor_mask = torch.ones(num_patches, dtype=torch.bool, device=patch_tokens.device)
    if num_anchors > 0:
        non_anchor_mask[anchor_indices] = False
    non_anchor_indices = torch.nonzero(non_anchor_mask, as_tuple=False).squeeze(-1)
    num_clusters = target - num_anchors

    if num_clusters > 0:
        centers = _initialize_centers(
            q_tokens=q_tokens,
            importance=importance,
            indices=non_anchor_indices,
            num_centers=num_clusters,
            lambda_diversity=cfg.lambda_diversity,
        )
        centers, local_assignments = _cluster_semantic_tokens(
            q_tokens=q_tokens,
            importance=importance,
            indices=non_anchor_indices,
            centers=centers,
            iterations=cfg.iterations,
            gamma=cfg.gamma,
            eps=cfg.eps,
        )
    else:
        centers = q_tokens.new_empty((0, q_tokens.shape[-1]))
        local_assignments = torch.empty(0, dtype=torch.long, device=patch_tokens.device)

    prototypes, masses, assignments, soft_positions = _aggregate_prototypes(
        patch_tokens=patch_tokens,
        q_tokens=q_tokens,
        importance=importance,
        anchor_indices=anchor_indices,
        non_anchor_indices=non_anchor_indices,
        centers=centers,
        local_assignments=local_assignments,
        positions=positions,
        temperature=cfg.temperature,
        lambda_importance=cfg.lambda_importance,
        eps=cfg.eps,
    )

    order = torch.arange(target, device=patch_tokens.device)
    if cfg.sort_by_position and soft_positions is not None and target > 1:
        order = _position_order(soft_positions)
        prototypes = prototypes[order]
        masses = masses[order]
        soft_positions = soft_positions[order]
        assignments = _remap_assignments(assignments, order)

    return SemReduceResult(
        sequence=None,
        patch_tokens=prototypes.to(dtype=patch_tokens.dtype),
        assignments=assignments,
        centers=centers,
        selected_classes=selected_classes,
        anchors=anchor_indices,
        masses=masses,
        soft_positions=soft_positions,
        prototype_order=order,
    )


def _semantic_response(
    patch_tokens: torch.Tensor,
    classifier_weight: torch.Tensor,
    cls_token: torch.Tensor | None,
    token_norm: Callable[[torch.Tensor], torch.Tensor],
    candidate_classes: int | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cls_token is None:
        cls_token = patch_tokens.mean(dim=0)

    norm_cls = token_norm(cls_token.unsqueeze(0)).to(dtype=torch.float32).squeeze(0)
    logits = norm_cls @ classifier_weight.T
    num_classes = int(classifier_weight.shape[0])
    if candidate_classes is None or int(candidate_classes) >= num_classes:
        selected = torch.arange(num_classes, device=patch_tokens.device)
    else:
        selected = torch.topk(logits, k=max(1, int(candidate_classes)), dim=-1).indices

    selected_weight = classifier_weight[selected]
    norm_patch = token_norm(patch_tokens).to(dtype=torch.float32)
    responses = norm_patch @ selected_weight.T
    mean = responses.mean(dim=0, keepdim=True)
    std = responses.std(dim=0, keepdim=True, unbiased=False)
    p_hat = (responses - mean) / (std + eps)
    q_tokens = F.normalize(p_hat, p=2, dim=-1, eps=eps)

    importance = p_hat.max(dim=-1).values - p_hat.mean(dim=-1)
    importance = (importance - importance.mean()) / (importance.std(unbiased=False) + eps)
    return q_tokens, p_hat, importance, selected


def _initialize_centers(
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    indices: torch.Tensor,
    num_centers: int,
    lambda_diversity: float,
) -> torch.Tensor:
    if num_centers <= 0:
        return q_tokens.new_empty((0, q_tokens.shape[-1]))
    if indices.numel() == 0:
        raise ValueError("no non-anchor tokens are available for clustering")

    q_subset = q_tokens[indices]
    u_subset = importance[indices]
    selected_local: list[int] = []
    first = int(torch.argmax(u_subset).item())
    selected_local.append(first)

    min_distance = 1.0 - (q_subset @ q_subset[first])
    for _ in range(1, num_centers):
        scores = u_subset + float(lambda_diversity) * min_distance
        if selected_local:
            scores = scores.clone()
            scores[torch.tensor(selected_local, device=scores.device)] = -torch.inf
        if bool(torch.isinf(scores).all()):
            candidate = int(torch.argmax(min_distance).item())
        else:
            candidate = int(torch.argmax(scores).item())
        selected_local.append(candidate)
        candidate_distance = 1.0 - (q_subset @ q_subset[candidate])
        min_distance = torch.minimum(min_distance, candidate_distance)

    return q_subset[selected_local].clone()


def _cluster_semantic_tokens(
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    indices: torch.Tensor,
    centers: torch.Tensor,
    iterations: int,
    gamma: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_subset = q_tokens[indices]
    u_subset = importance[indices]
    assignments = torch.zeros(q_subset.shape[0], dtype=torch.long, device=q_subset.device)

    for _ in range(max(0, int(iterations))):
        assignments = (q_subset @ centers.T).argmax(dim=-1)
        updated = []
        for center_index in range(int(centers.shape[0])):
            member_mask = assignments == center_index
            if bool(member_mask.any()):
                weights = torch.exp(float(gamma) * u_subset[member_mask]).unsqueeze(-1)
                weighted = (weights * q_subset[member_mask]).sum(dim=0, keepdim=True)
                updated.append(F.normalize(weighted, p=2, dim=-1, eps=eps).squeeze(0))
            else:
                updated.append(_reinitialize_empty_center(q_subset, u_subset, centers, center_index))
        centers = torch.stack(updated, dim=0)

    assignments = (q_subset @ centers.T).argmax(dim=-1)
    assignments = _repair_empty_assignments(q_subset, u_subset, centers, assignments)
    return centers, assignments


def _reinitialize_empty_center(
    q_subset: torch.Tensor,
    u_subset: torch.Tensor,
    centers: torch.Tensor,
    center_index: int,
) -> torch.Tensor:
    if centers.shape[0] == 1:
        return q_subset[int(torch.argmax(u_subset).item())]
    other_indices = [index for index in range(int(centers.shape[0])) if index != center_index]
    other_centers = centers[other_indices]
    diversity = (1.0 - q_subset @ other_centers.T).min(dim=-1).values
    candidate = torch.argmax(u_subset + diversity)
    return q_subset[candidate]


def _repair_empty_assignments(
    q_subset: torch.Tensor,
    u_subset: torch.Tensor,
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
        donor_mask = counts[repaired] > 1
        donor_indices = torch.nonzero(donor_mask, as_tuple=False).flatten()
        if donor_indices.numel() == 0:
            break
        current_similarity = (q_subset * centers[repaired]).sum(dim=-1)
        donor_scores = u_subset - current_similarity
        moved_local = donor_indices[torch.argmax(donor_scores[donor_indices])]
        counts[repaired[moved_local]] -= 1
        repaired[moved_local] = int(empty_center)
        counts[empty_center] += 1
    return repaired


def _aggregate_prototypes(
    patch_tokens: torch.Tensor,
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    anchor_indices: torch.Tensor,
    non_anchor_indices: torch.Tensor,
    centers: torch.Tensor,
    local_assignments: torch.Tensor,
    positions: torch.Tensor | None,
    temperature: float,
    lambda_importance: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    del eps
    prototypes: list[torch.Tensor] = []
    masses: list[torch.Tensor] = []
    soft_positions: list[torch.Tensor] = []
    assignments = torch.empty(patch_tokens.shape[0], dtype=torch.long, device=patch_tokens.device)
    prepared_positions = _prepare_positions(positions, int(patch_tokens.shape[0]), patch_tokens.device)

    for output_index, patch_index in enumerate(anchor_indices.tolist()):
        prototypes.append(patch_tokens[patch_index])
        masses.append(torch.tensor(1, dtype=torch.long, device=patch_tokens.device))
        assignments[patch_index] = output_index
        if prepared_positions is not None:
            soft_positions.append(prepared_positions[patch_index])

    base = len(prototypes)
    for center_index in range(int(centers.shape[0])):
        member_mask = local_assignments == center_index
        if not bool(member_mask.any()):
            raise RuntimeError("semantic clustering produced an empty cluster after repair")
        member_indices = non_anchor_indices[member_mask]

        semantic_scores = q_tokens[member_indices] @ centers[center_index]
        scores = semantic_scores + float(lambda_importance) * importance[member_indices]
        weights = torch.softmax(scores / float(temperature), dim=0).to(dtype=patch_tokens.dtype)
        prototype = (patch_tokens[member_indices] * weights.unsqueeze(-1)).sum(dim=0)
        output_index = base + center_index
        prototypes.append(prototype)
        masses.append(torch.tensor(member_indices.numel(), dtype=torch.long, device=patch_tokens.device))
        assignments[member_indices] = output_index
        if prepared_positions is not None:
            pos_weights = weights.to(dtype=prepared_positions.dtype)
            soft_positions.append((prepared_positions[member_indices] * pos_weights.unsqueeze(-1)).sum(dim=0))

    stacked_positions = torch.stack(soft_positions, dim=0) if soft_positions else None
    return torch.stack(prototypes, dim=0), torch.stack(masses), assignments, stacked_positions


def _prepare_positions(
    positions: torch.Tensor | None,
    num_patches: int,
    device: torch.device,
) -> torch.Tensor | None:
    if positions is None:
        side = math.isqrt(num_patches)
        if side * side == num_patches:
            rows = torch.arange(side, device=device, dtype=torch.float32)
            cols = torch.arange(side, device=device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
            return torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
        row = torch.zeros(num_patches, device=device, dtype=torch.float32)
        col = torch.arange(num_patches, device=device, dtype=torch.float32)
        return torch.stack([row, col], dim=-1)

    if positions.shape != (num_patches, 2):
        raise ValueError(f"positions must have shape [{num_patches}, 2], got {tuple(positions.shape)}")
    return positions.to(device=device, dtype=torch.float32)


def _position_order(positions: torch.Tensor) -> torch.Tensor:
    width = positions[:, 1].max().clamp(min=0.0) + 1.0
    return torch.argsort(positions[:, 0] * width + positions[:, 1], stable=True)


def _remap_assignments(assignments: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return inverse[assignments]


def _topk_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    return torch.topk(values, k=k, dim=0).indices


def _make_norm(token_norm: TokenNorm) -> Callable[[torch.Tensor], torch.Tensor]:
    if token_norm is None:
        return lambda values: F.layer_norm(values.float(), values.shape[-1:])
    return token_norm


def _classifier_weight(classifier: torch.Tensor | nn.Module, device: torch.device) -> torch.Tensor:
    weight = classifier
    if isinstance(classifier, nn.Module):
        if not hasattr(classifier, "weight"):
            raise ValueError("classifier module must expose a .weight tensor")
        weight = classifier.weight
    if not isinstance(weight, torch.Tensor):
        raise TypeError("classifier must be a Tensor or nn.Module with .weight")
    if weight.ndim != 2:
        raise ValueError(f"classifier weight must have shape [K, D], got {tuple(weight.shape)}")
    return weight.to(device=device, dtype=torch.float32)


def _resolve_config(
    config: SemReduceConfig | None,
    overrides: dict[str, object],
) -> SemReduceConfig:
    if config is None:
        if "num_prototypes" not in overrides:
            raise ValueError("num_prototypes is required when config is not provided")
        return SemReduceConfig(**overrides)  # type: ignore[arg-type]
    if not overrides:
        return config
    values = {**config.__dict__, **overrides}
    return SemReduceConfig(**values)


def _validate_config(cfg: SemReduceConfig) -> None:
    if cfg.num_prototypes <= 0:
        raise ValueError("num_prototypes must be positive")
    if cfg.temperature <= 0:
        raise ValueError("temperature must be positive")
    if cfg.candidate_classes is not None and cfg.candidate_classes <= 0:
        raise ValueError("candidate_classes must be positive or None")


def _stack_results(results: list[SemReduceResult], include_sequence: bool) -> SemReduceResult:
    sequence = None
    if include_sequence:
        sequence = torch.stack([result.sequence for result in results if result.sequence is not None], dim=0)
    soft_positions = None
    if results[0].soft_positions is not None:
        soft_positions = torch.stack([result.soft_positions for result in results], dim=0)
    return SemReduceResult(
        sequence=sequence,
        patch_tokens=torch.stack([result.patch_tokens for result in results], dim=0),
        assignments=torch.stack([result.assignments for result in results], dim=0),
        centers=torch.stack([result.centers for result in results], dim=0),
        selected_classes=torch.stack([result.selected_classes for result in results], dim=0),
        anchors=torch.stack([result.anchors for result in results], dim=0),
        masses=torch.stack([result.masses for result in results], dim=0),
        soft_positions=soft_positions,
        prototype_order=torch.stack([result.prototype_order for result in results], dim=0),
    )
