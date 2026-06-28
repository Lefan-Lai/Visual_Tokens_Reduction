from __future__ import annotations

import heapq
import itertools
import math
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
    _prepare_positions,
    _semantic_response,
    _stack_results,
)


@dataclass(frozen=True)
class FloodSemReduceConfig:
    """Configuration for Flood-SemReduce.

    Unlike Early-SemReduce, Flood-SemReduce does not take a fixed target token
    count. The output count is the number of final semantic-admissible regions.
    """

    candidate_classes: int | None = 64
    eps: float = 1e-6
    sort_by_position: bool = True


@dataclass
class _Region:
    id: int
    members: list[int]
    center: torch.Tensor
    importance: float
    position: torch.Tensor
    active: bool = True


class FloodSemReduce(nn.Module):
    """Training-free semantic-admissible region growing token reduction."""

    def __init__(self, config: FloodSemReduceConfig, token_norm: TokenNorm = None) -> None:
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
            raise ValueError("FloodSemReduce currently expects the CLS token at index 0")
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
        return _stack_dynamic_results(per_batch, include_sequence=True)

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
        reduced = flood_semreduce(
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


def flood_semreduce(
    patch_tokens: torch.Tensor,
    classifier: torch.Tensor | nn.Module,
    cls_token: torch.Tensor | None = None,
    config: FloodSemReduceConfig | None = None,
    token_norm: TokenNorm = None,
    positions: torch.Tensor | None = None,
    **config_overrides: object,
) -> SemReduceResult:
    """Reduce patch tokens by semantic-admissible region growing."""

    cfg = _resolve_flood_config(config, config_overrides)
    if patch_tokens.ndim == 2:
        return _flood_semreduce_single(patch_tokens, classifier, cls_token, cfg, token_norm, positions)
    if patch_tokens.ndim != 3:
        raise ValueError(
            f"Expected patch_tokens with shape [N, D] or [B, N, D], got {tuple(patch_tokens.shape)}"
        )

    per_batch = []
    for batch_index, batch_tokens in enumerate(patch_tokens):
        batch_cls = None if cls_token is None else cls_token[batch_index]
        batch_positions = None if positions is None else positions[batch_index]
        per_batch.append(
            _flood_semreduce_single(
                batch_tokens,
                classifier,
                batch_cls,
                cfg,
                token_norm,
                batch_positions,
            )
        )
    return _stack_dynamic_results(per_batch, include_sequence=False)


def _flood_semreduce_single(
    patch_tokens: torch.Tensor,
    classifier: torch.Tensor | nn.Module,
    cls_token: torch.Tensor | None,
    cfg: FloodSemReduceConfig,
    token_norm: TokenNorm,
    positions: torch.Tensor | None,
) -> SemReduceResult:
    _validate_flood_config(cfg)
    if patch_tokens.ndim != 2:
        raise ValueError(f"Expected [N, D] patch tokens, got {tuple(patch_tokens.shape)}")
    if patch_tokens.shape[0] == 0:
        raise ValueError("cannot reduce an empty patch-token sequence")

    num_patches = int(patch_tokens.shape[0])
    norm_fn = _make_norm(token_norm)
    weight = _classifier_weight(classifier, patch_tokens.device)
    q_tokens, _p_hat, importance, selected_classes = _semantic_response(
        patch_tokens=patch_tokens,
        classifier_weight=weight,
        cls_token=cls_token,
        token_norm=norm_fn,
        candidate_classes=cfg.candidate_classes,
        eps=cfg.eps,
    )
    prepared_positions = _prepare_positions(positions, num_patches, patch_tokens.device)
    if prepared_positions is None:
        raise RuntimeError("Flood-SemReduce requires patch positions")

    patch_neighbors = _build_patch_neighbors(prepared_positions)
    initial_edges = _unique_edges(patch_neighbors)
    importance01 = _minmax01(importance, cfg.eps)

    regions: dict[int, _Region] = {
        index: _Region(
            id=index,
            members=[index],
            center=q_tokens[index],
            importance=float(importance[index].item()),
            position=prepared_positions[index],
        )
        for index in range(num_patches)
    }
    patch_to_region = torch.arange(num_patches, dtype=torch.long, device=patch_tokens.device)

    if not initial_edges:
        return _make_result_from_regions(
            patch_tokens=patch_tokens,
            q_tokens=q_tokens,
            importance01=importance01,
            selected_classes=selected_classes,
            regions=list(regions.values()),
            sort_by_position=cfg.sort_by_position,
        )

    similarities = torch.stack([(q_tokens[i] * q_tokens[j]).sum() for i, j in initial_edges])
    differences = torch.stack([torch.abs(importance[i] - importance[j]) for i, j in initial_edges])
    similarity_threshold = _adaptive_otsu_threshold(similarities)
    difference_threshold = _adaptive_otsu_threshold(differences)

    heap: list[tuple[float, float, int, int, int]] = []
    counter = itertools.count()
    for left, right in initial_edges:
        _push_if_admissible(
            heap=heap,
            counter=counter,
            regions=regions,
            left_id=left,
            right_id=right,
            q_tokens=q_tokens,
            importance=importance,
            patch_neighbors=patch_neighbors,
            patch_to_region=patch_to_region,
            similarity_threshold=similarity_threshold,
            difference_threshold=difference_threshold,
        )

    next_region_id = num_patches
    while heap:
        _priority_similarity, _priority_difference, _entry_id, left_id, right_id = heapq.heappop(heap)
        left = regions.get(left_id)
        right = regions.get(right_id)
        if left is None or right is None or not left.active or not right.active:
            continue

        metrics = _region_pair_metrics(
            left,
            right,
            q_tokens=q_tokens,
            importance=importance,
            patch_neighbors=patch_neighbors,
            patch_to_region=patch_to_region,
        )
        if metrics is None or not _is_admissible(metrics, similarity_threshold, difference_threshold):
            continue

        merged = _merge_regions(
            next_region_id,
            left,
            right,
            q_tokens=q_tokens,
            importance=importance,
            importance01=importance01,
            positions=prepared_positions,
            eps=cfg.eps,
        )
        if not _passes_consistency(left, right, merged, similarity_threshold, difference_threshold):
            continue

        left.active = False
        right.active = False
        regions[next_region_id] = merged
        member_index = torch.tensor(merged.members, dtype=torch.long, device=patch_tokens.device)
        patch_to_region[member_index] = next_region_id
        next_region_id += 1

        for neighbor_id in _active_neighbor_region_ids(
            merged,
            patch_neighbors=patch_neighbors,
            patch_to_region=patch_to_region,
            regions=regions,
        ):
            _push_if_admissible(
                heap=heap,
                counter=counter,
                regions=regions,
                left_id=merged.id,
                right_id=neighbor_id,
                q_tokens=q_tokens,
                importance=importance,
                patch_neighbors=patch_neighbors,
                patch_to_region=patch_to_region,
                similarity_threshold=similarity_threshold,
                difference_threshold=difference_threshold,
            )

    final_regions = [region for region in regions.values() if region.active]
    return _make_result_from_regions(
        patch_tokens=patch_tokens,
        q_tokens=q_tokens,
        importance01=importance01,
        selected_classes=selected_classes,
        regions=final_regions,
        sort_by_position=cfg.sort_by_position,
    )


def _build_patch_neighbors(positions: torch.Tensor) -> list[set[int]]:
    keys = [(int(round(float(pos[0].item()))), int(round(float(pos[1].item())))) for pos in positions]
    coordinate_to_index = {key: index for index, key in enumerate(keys)}
    neighbors: list[set[int]] = [set() for _ in keys]
    for index, (row, col) in enumerate(keys):
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = coordinate_to_index.get((row + delta_row, col + delta_col))
            if neighbor is not None:
                neighbors[index].add(neighbor)
    return neighbors


def _unique_edges(neighbors: list[set[int]]) -> list[tuple[int, int]]:
    edges = []
    for left, right_values in enumerate(neighbors):
        for right in right_values:
            if left < right:
                edges.append((left, right))
    return edges


def _adaptive_otsu_threshold(values: torch.Tensor) -> float:
    flat = values.detach().float().flatten()
    if flat.numel() == 0:
        return 0.0
    if flat.numel() == 1:
        return float(flat.item())

    sorted_values = torch.sort(flat).values
    if float((sorted_values[-1] - sorted_values[0]).abs().item()) <= 1e-12:
        return float(sorted_values.median().item())

    prefix_sum = torch.cumsum(sorted_values, dim=0)
    prefix_sq_sum = torch.cumsum(sorted_values * sorted_values, dim=0)
    total_sum = prefix_sum[-1]
    total_sq_sum = prefix_sq_sum[-1]
    count = torch.arange(1, flat.numel(), device=flat.device, dtype=torch.float32)
    left_count = count
    right_count = float(flat.numel()) - count

    left_sum = prefix_sum[:-1]
    right_sum = total_sum - left_sum
    left_sq_sum = prefix_sq_sum[:-1]
    right_sq_sum = total_sq_sum - left_sq_sum

    left_mean = left_sum / left_count
    right_mean = right_sum / right_count
    left_var = left_sq_sum / left_count - left_mean * left_mean
    right_var = right_sq_sum / right_count - right_mean * right_mean
    within = left_count * left_var.clamp_min(0.0) + right_count * right_var.clamp_min(0.0)

    best = int(torch.argmin(within).item())
    threshold = 0.5 * (sorted_values[best] + sorted_values[best + 1])
    if not math.isfinite(float(threshold.item())):
        return float(sorted_values.median().item())
    return float(threshold.item())


def _minmax01(values: torch.Tensor, eps: float) -> torch.Tensor:
    minimum = values.min()
    maximum = values.max()
    return (values - minimum) / (maximum - minimum + eps)


def _region_pair_metrics(
    left: _Region,
    right: _Region,
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    patch_neighbors: list[set[int]],
    patch_to_region: torch.Tensor,
) -> tuple[float, float, float] | None:
    boundary_pairs = []
    right_id = int(right.id)
    for patch_index in left.members:
        for neighbor in patch_neighbors[patch_index]:
            if int(patch_to_region[neighbor].item()) == right_id:
                boundary_pairs.append((patch_index, neighbor))
    if not boundary_pairs:
        return None

    s_reg = float((left.center * right.center).sum().item())
    s_boundary_values = [(q_tokens[i] * q_tokens[j]).sum() for i, j in boundary_pairs]
    d_boundary_values = [torch.abs(importance[i] - importance[j]) for i, j in boundary_pairs]
    s_boundary = float(torch.stack(s_boundary_values).mean().item())
    d_boundary = float(torch.stack(d_boundary_values).mean().item())
    return s_reg, s_boundary, d_boundary


def _is_admissible(
    metrics: tuple[float, float, float],
    similarity_threshold: float,
    difference_threshold: float,
) -> bool:
    s_reg, s_boundary, d_boundary = metrics
    return (
        s_reg >= similarity_threshold
        and s_boundary >= similarity_threshold
        and d_boundary <= difference_threshold
    )


def _push_if_admissible(
    heap: list[tuple[float, float, int, int, int]],
    counter: itertools.count,
    regions: dict[int, _Region],
    left_id: int,
    right_id: int,
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    patch_neighbors: list[set[int]],
    patch_to_region: torch.Tensor,
    similarity_threshold: float,
    difference_threshold: float,
) -> None:
    if left_id == right_id:
        return
    left = regions.get(int(left_id))
    right = regions.get(int(right_id))
    if left is None or right is None or not left.active or not right.active:
        return
    metrics = _region_pair_metrics(
        left,
        right,
        q_tokens=q_tokens,
        importance=importance,
        patch_neighbors=patch_neighbors,
        patch_to_region=patch_to_region,
    )
    if metrics is None or not _is_admissible(metrics, similarity_threshold, difference_threshold):
        return
    s_reg, s_boundary, d_boundary = metrics
    eta = min(s_reg, s_boundary)
    heapq.heappush(heap, (-eta, d_boundary, next(counter), int(left_id), int(right_id)))


def _merge_regions(
    region_id: int,
    left: _Region,
    right: _Region,
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    importance01: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> _Region:
    members = sorted(left.members + right.members)
    member_index = torch.tensor(members, dtype=torch.long, device=q_tokens.device)
    weights = torch.softmax(importance01[member_index], dim=0)
    weighted_center = (q_tokens[member_index] * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
    center = F.normalize(weighted_center, p=2, dim=-1, eps=eps).squeeze(0)
    position = (positions[member_index] * weights.unsqueeze(-1).to(dtype=positions.dtype)).sum(dim=0)
    return _Region(
        id=region_id,
        members=members,
        center=center,
        importance=float(importance[member_index].mean().item()),
        position=position,
    )


def _passes_consistency(
    left: _Region,
    right: _Region,
    merged: _Region,
    similarity_threshold: float,
    difference_threshold: float,
) -> bool:
    left_similarity = float((left.center * merged.center).sum().item())
    right_similarity = float((right.center * merged.center).sum().item())
    left_difference = abs(left.importance - merged.importance)
    right_difference = abs(right.importance - merged.importance)
    return (
        left_similarity >= similarity_threshold
        and right_similarity >= similarity_threshold
        and left_difference <= difference_threshold
        and right_difference <= difference_threshold
    )


def _active_neighbor_region_ids(
    region: _Region,
    patch_neighbors: list[set[int]],
    patch_to_region: torch.Tensor,
    regions: dict[int, _Region],
) -> list[int]:
    neighbor_ids = set()
    for patch_index in region.members:
        for neighbor in patch_neighbors[patch_index]:
            neighbor_id = int(patch_to_region[neighbor].item())
            if neighbor_id == region.id:
                continue
            neighbor_region = regions.get(neighbor_id)
            if neighbor_region is not None and neighbor_region.active:
                neighbor_ids.add(neighbor_id)
    return sorted(neighbor_ids)


def _make_result_from_regions(
    patch_tokens: torch.Tensor,
    q_tokens: torch.Tensor,
    importance01: torch.Tensor,
    selected_classes: torch.Tensor,
    regions: list[_Region],
    sort_by_position: bool,
) -> SemReduceResult:
    if sort_by_position:
        regions = sorted(regions, key=lambda region: (float(region.position[0]), float(region.position[1])))
    else:
        regions = sorted(regions, key=lambda region: region.id)

    prototypes = []
    centers = []
    masses = []
    soft_positions = []
    assignments = torch.empty(patch_tokens.shape[0], dtype=torch.long, device=patch_tokens.device)
    prototype_order = []

    for output_index, region in enumerate(regions):
        member_index = torch.tensor(region.members, dtype=torch.long, device=patch_tokens.device)
        semantic_scores = (q_tokens[member_index] @ region.center + 1.0) * 0.5
        scores = semantic_scores + importance01[member_index]
        weights = torch.softmax(scores, dim=0).to(dtype=patch_tokens.dtype)
        prototype = (patch_tokens[member_index] * weights.unsqueeze(-1)).sum(dim=0)
        prototypes.append(prototype)
        centers.append(region.center)
        masses.append(torch.tensor(member_index.numel(), dtype=torch.long, device=patch_tokens.device))
        soft_positions.append(region.position)
        assignments[member_index] = output_index
        prototype_order.append(region.id)

    return SemReduceResult(
        sequence=None,
        patch_tokens=torch.stack(prototypes, dim=0).to(dtype=patch_tokens.dtype),
        assignments=assignments,
        centers=torch.stack(centers, dim=0),
        selected_classes=selected_classes,
        anchors=torch.empty(0, dtype=torch.long, device=patch_tokens.device),
        masses=torch.stack(masses),
        soft_positions=torch.stack(soft_positions, dim=0),
        prototype_order=torch.tensor(prototype_order, dtype=torch.long, device=patch_tokens.device),
    )


def _resolve_flood_config(
    config: FloodSemReduceConfig | None,
    overrides: dict[str, object],
) -> FloodSemReduceConfig:
    if config is None:
        return FloodSemReduceConfig(**overrides)  # type: ignore[arg-type]
    if not overrides:
        return config
    values = {**config.__dict__, **overrides}
    return FloodSemReduceConfig(**values)


def _validate_flood_config(cfg: FloodSemReduceConfig) -> None:
    if cfg.eps <= 0:
        raise ValueError("eps must be positive")
    if cfg.candidate_classes is not None and cfg.candidate_classes <= 0:
        raise ValueError("candidate_classes must be positive or None")


def _stack_dynamic_results(results: list[SemReduceResult], include_sequence: bool) -> SemReduceResult:
    lengths = {int(result.patch_tokens.shape[-2]) for result in results}
    if len(lengths) != 1:
        raise ValueError(
            "Flood-SemReduce produced different token counts across the batch; "
            "run batch size 1 or pad the dynamic outputs before stacking"
        )
    return _stack_results(results, include_sequence=include_sequence)
