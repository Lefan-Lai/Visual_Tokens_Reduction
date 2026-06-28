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
    _prepare_positions,
    _semantic_response,
    _stack_results,
)


@dataclass(frozen=True)
class FastFloodSemReduceConfig:
    """Configuration for Fast Flood-SemReduce.

    The algorithm has no fixed token budget. It builds a 4-neighbor patch graph,
    keeps semantic-admissible edges, and turns connected components into reduced
    tokens.
    """

    candidate_classes: int | None = 64
    eps: float = 1e-6
    sort_by_position: bool = True
    strict_consistency: bool = True
    consistency_ratio: float = 1.0


@dataclass
class _Component:
    members: list[int]
    center: torch.Tensor
    importance: float


class FastFloodSemReduce(nn.Module):
    """Training-free semantic-admissible connected-component token reduction."""

    def __init__(self, config: FastFloodSemReduceConfig, token_norm: TokenNorm = None) -> None:
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
            raise ValueError("FastFloodSemReduce currently expects the CLS token at index 0")
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
        reduced = fast_flood_semreduce(
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


def fast_flood_semreduce(
    patch_tokens: torch.Tensor,
    classifier: torch.Tensor | nn.Module,
    cls_token: torch.Tensor | None = None,
    config: FastFloodSemReduceConfig | None = None,
    token_norm: TokenNorm = None,
    positions: torch.Tensor | None = None,
    **config_overrides: object,
) -> SemReduceResult:
    """Reduce patch tokens with Fast Flood-SemReduce."""

    cfg = _resolve_fast_flood_config(config, config_overrides)
    if patch_tokens.ndim == 2:
        return _fast_flood_semreduce_single(patch_tokens, classifier, cls_token, cfg, token_norm, positions)
    if patch_tokens.ndim != 3:
        raise ValueError(
            f"Expected patch_tokens with shape [N, D] or [B, N, D], got {tuple(patch_tokens.shape)}"
        )

    per_batch = []
    for batch_index, batch_tokens in enumerate(patch_tokens):
        batch_cls = None if cls_token is None else cls_token[batch_index]
        batch_positions = None if positions is None else positions[batch_index]
        per_batch.append(
            _fast_flood_semreduce_single(
                batch_tokens,
                classifier,
                batch_cls,
                cfg,
                token_norm,
                batch_positions,
            )
        )
    return _stack_dynamic_results(per_batch, include_sequence=False)


def _fast_flood_semreduce_single(
    patch_tokens: torch.Tensor,
    classifier: torch.Tensor | nn.Module,
    cls_token: torch.Tensor | None,
    cfg: FastFloodSemReduceConfig,
    token_norm: TokenNorm,
    positions: torch.Tensor | None,
) -> SemReduceResult:
    _validate_fast_flood_config(cfg)
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
    importance01 = _minmax01(importance, cfg.eps)
    prepared_positions = _prepare_positions(positions, num_patches, patch_tokens.device)
    if prepared_positions is None:
        raise RuntimeError("Fast Flood-SemReduce requires patch positions")

    edges = _build_grid_edges(prepared_positions)
    if not edges:
        components = [
            _make_component([index], q_tokens, importance, importance01, cfg.eps)
            for index in range(num_patches)
        ]
    else:
        edge_tensor = torch.tensor(edges, dtype=torch.long, device=patch_tokens.device)
        similarities = (q_tokens[edge_tensor[:, 0]] * q_tokens[edge_tensor[:, 1]]).sum(dim=-1)
        differences = torch.abs(importance[edge_tensor[:, 0]] - importance[edge_tensor[:, 1]])
        similarity_threshold = _adaptive_otsu_threshold(similarities)
        difference_threshold = _adaptive_otsu_threshold(differences)
        legal_edges = [
            edge
            for edge, similarity, difference in zip(edges, similarities.tolist(), differences.tolist())
            if similarity >= similarity_threshold and difference <= difference_threshold
        ]
        components = _connected_components(num_patches, legal_edges, q_tokens, importance, importance01, cfg.eps)
        components = _repair_inconsistent_components(
            components=components,
            legal_edges=legal_edges,
            q_tokens=q_tokens,
            importance=importance,
            importance01=importance01,
            similarity_threshold=similarity_threshold,
            difference_threshold=difference_threshold,
            consistency_ratio=cfg.consistency_ratio,
            strict=cfg.strict_consistency,
            eps=cfg.eps,
        )

    return _make_result_from_components(
        patch_tokens=patch_tokens,
        q_tokens=q_tokens,
        importance01=importance01,
        positions=prepared_positions,
        selected_classes=selected_classes,
        components=components,
        sort_by_position=cfg.sort_by_position,
    )


def _build_grid_edges(positions: torch.Tensor) -> list[tuple[int, int]]:
    keys = [(int(round(float(pos[0].item()))), int(round(float(pos[1].item())))) for pos in positions]
    coordinate_to_index = {key: index for index, key in enumerate(keys)}
    edges = []
    for index, (row, col) in enumerate(keys):
        for delta_row, delta_col in ((1, 0), (0, 1)):
            neighbor = coordinate_to_index.get((row + delta_row, col + delta_col))
            if neighbor is not None:
                edges.append((index, neighbor))
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
    return float(threshold.item())


def _connected_components(
    num_nodes: int,
    edges: list[tuple[int, int]],
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    importance01: torch.Tensor,
    eps: float,
) -> list[_Component]:
    parent = list(range(num_nodes))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in edges:
        union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(num_nodes):
        groups.setdefault(find(index), []).append(index)

    return [
        _make_component(sorted(members), q_tokens, importance, importance01, eps)
        for members in groups.values()
    ]


def _make_component(
    members: list[int],
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    importance01: torch.Tensor,
    eps: float,
) -> _Component:
    index = torch.tensor(members, dtype=torch.long, device=q_tokens.device)
    weights = torch.softmax(importance01[index], dim=0)
    center = F.normalize(
        (q_tokens[index] * weights.unsqueeze(-1)).sum(dim=0, keepdim=True),
        p=2,
        dim=-1,
        eps=eps,
    ).squeeze(0)
    return _Component(
        members=members,
        center=center,
        importance=float(importance[index].mean().item()),
    )


def _repair_inconsistent_components(
    components: list[_Component],
    legal_edges: list[tuple[int, int]],
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    importance01: torch.Tensor,
    similarity_threshold: float,
    difference_threshold: float,
    consistency_ratio: float,
    strict: bool,
    eps: float,
) -> list[_Component]:
    repaired = []
    legal_edge_set = {tuple(sorted(edge)) for edge in legal_edges}
    for component in components:
        if _component_is_consistent(
            component,
            q_tokens=q_tokens,
            importance=importance,
            similarity_threshold=similarity_threshold,
            difference_threshold=difference_threshold,
            ratio=consistency_ratio,
            strict=strict,
        ):
            repaired.append(component)
            continue

        good_members = _consistent_member_mask(
            component,
            q_tokens=q_tokens,
            importance=importance,
            similarity_threshold=similarity_threshold,
            difference_threshold=difference_threshold,
        )
        repaired_edges = []
        component_set = set(component.members)
        for left, right in legal_edge_set:
            if left not in component_set or right not in component_set:
                continue
            if good_members.get(left, False) and good_members.get(right, False):
                repaired_edges.append((left, right))
        repaired.extend(
            _connected_components_for_members(
                component.members,
                repaired_edges,
                q_tokens=q_tokens,
                importance=importance,
                importance01=importance01,
                eps=eps,
            )
        )
    return repaired


def _component_is_consistent(
    component: _Component,
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    similarity_threshold: float,
    difference_threshold: float,
    ratio: float,
    strict: bool,
) -> bool:
    good = _consistent_member_mask(
        component,
        q_tokens=q_tokens,
        importance=importance,
        similarity_threshold=similarity_threshold,
        difference_threshold=difference_threshold,
    )
    good_count = sum(1 for value in good.values() if value)
    if strict:
        return good_count == len(component.members)
    return good_count / max(1, len(component.members)) >= ratio


def _consistent_member_mask(
    component: _Component,
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    similarity_threshold: float,
    difference_threshold: float,
) -> dict[int, bool]:
    index = torch.tensor(component.members, dtype=torch.long, device=q_tokens.device)
    similarities = q_tokens[index] @ component.center
    differences = torch.abs(importance[index] - component.importance)
    flags = (similarities >= similarity_threshold) & (differences <= difference_threshold)
    return {member: bool(flag) for member, flag in zip(component.members, flags.tolist())}


def _connected_components_for_members(
    members: list[int],
    edges: list[tuple[int, int]],
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    importance01: torch.Tensor,
    eps: float,
) -> list[_Component]:
    member_set = set(members)
    if not edges:
        return [_make_component([member], q_tokens, importance, importance01, eps) for member in sorted(member_set)]
    local_index = {member: idx for idx, member in enumerate(sorted(member_set))}
    local_edges = [(local_index[left], local_index[right]) for left, right in edges]
    local_components = _connected_components(
        len(local_index),
        local_edges,
        q_tokens=torch.stack([q_tokens[member] for member in sorted(member_set)], dim=0),
        importance=torch.stack([importance[member] for member in sorted(member_set)], dim=0),
        importance01=torch.stack([importance01[member] for member in sorted(member_set)], dim=0),
        eps=eps,
    )
    reverse = sorted(member_set)
    return [
        _make_component(
            [reverse[local_member] for local_member in component.members],
            q_tokens,
            importance,
            importance01,
            eps,
        )
        for component in local_components
    ]


def _make_result_from_components(
    patch_tokens: torch.Tensor,
    q_tokens: torch.Tensor,
    importance01: torch.Tensor,
    positions: torch.Tensor,
    selected_classes: torch.Tensor,
    components: list[_Component],
    sort_by_position: bool,
) -> SemReduceResult:
    records = []
    for component in components:
        member_index = torch.tensor(component.members, dtype=torch.long, device=patch_tokens.device)
        semantic_scores = (q_tokens[member_index] @ component.center + 1.0) * 0.5
        scores = semantic_scores + importance01[member_index]
        weights = torch.softmax(scores, dim=0).to(dtype=patch_tokens.dtype)
        prototype = (patch_tokens[member_index] * weights.unsqueeze(-1)).sum(dim=0)
        position = (positions[member_index] * weights.unsqueeze(-1).to(dtype=positions.dtype)).sum(dim=0)
        records.append((position, prototype, component, member_index))

    if sort_by_position:
        records.sort(key=lambda item: (float(item[0][0]), float(item[0][1])))

    prototypes = []
    centers = []
    masses = []
    soft_positions = []
    assignments = torch.empty(patch_tokens.shape[0], dtype=torch.long, device=patch_tokens.device)
    prototype_order = []

    for output_index, (position, prototype, component, member_index) in enumerate(records):
        prototypes.append(prototype)
        centers.append(component.center)
        masses.append(torch.tensor(member_index.numel(), dtype=torch.long, device=patch_tokens.device))
        soft_positions.append(position)
        assignments[member_index] = output_index
        prototype_order.append(min(component.members))

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


def _minmax01(values: torch.Tensor, eps: float) -> torch.Tensor:
    minimum = values.min()
    maximum = values.max()
    return (values - minimum) / (maximum - minimum + eps)


def _resolve_fast_flood_config(
    config: FastFloodSemReduceConfig | None,
    overrides: dict[str, object],
) -> FastFloodSemReduceConfig:
    if config is None:
        return FastFloodSemReduceConfig(**overrides)  # type: ignore[arg-type]
    if not overrides:
        return config
    values = {**config.__dict__, **overrides}
    return FastFloodSemReduceConfig(**values)


def _validate_fast_flood_config(cfg: FastFloodSemReduceConfig) -> None:
    if cfg.eps <= 0:
        raise ValueError("eps must be positive")
    if cfg.candidate_classes is not None and cfg.candidate_classes <= 0:
        raise ValueError("candidate_classes must be positive or None")
    if not 0 < cfg.consistency_ratio <= 1:
        raise ValueError("consistency_ratio must be in (0, 1]")


def _stack_dynamic_results(results: list[SemReduceResult], include_sequence: bool) -> SemReduceResult:
    lengths = {int(result.patch_tokens.shape[-2]) for result in results}
    if len(lengths) != 1:
        raise ValueError(
            "Fast Flood-SemReduce produced different token counts across the batch; "
            "run batch size 1 or pad the dynamic outputs before stacking"
        )
    return _stack_results(results, include_sequence=include_sequence)
