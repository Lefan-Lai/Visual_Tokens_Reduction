from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


TokenNorm = Callable[[torch.Tensor], torch.Tensor] | None


@dataclass(frozen=True)
class ForesightConfig:
    k_min: int = 32
    k_max: int = 128
    k_text: int = 64
    rho: float = 0.90
    eps: float = 1e-6
    temperature: float = 1.0
    lambda_importance: float = 1.0
    sort_by_position: bool = True


@dataclass
class ForesightResult:
    patch_tokens: torch.Tensor
    assignments: torch.Tensor
    centers: torch.Tensor
    hypothesis_indices: torch.Tensor
    seed_indices: torch.Tensor
    masses: torch.Tensor
    soft_positions: torch.Tensor | None
    prototype_order: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_scores: torch.Tensor
    image_confidence: torch.Tensor
    patch_support: torch.Tensor
    k_min: int
    k_max: int
    k_budget: int
    k_rho: int
    k_eff: int
    iterations_used: int
    final_assignment_change_rate: float | None
    image_hypothesis_count: int
    text_hypothesis_count: int


def foresight_reduce(
    patch_tokens: torch.Tensor,
    text_hypotheses: torch.Tensor | None = None,
    config: ForesightConfig | None = None,
    token_norm: TokenNorm = None,
    positions: torch.Tensor | None = None,
) -> ForesightResult:
    cfg = config or ForesightConfig()
    _validate_config(cfg)
    if patch_tokens.ndim == 2:
        return _foresight_single(patch_tokens, text_hypotheses, cfg, token_norm, positions)
    if patch_tokens.ndim != 3:
        raise ValueError(f"patch_tokens must have shape [N, D] or [B, N, D], got {tuple(patch_tokens.shape)}")

    results = []
    for batch_index, batch_tokens in enumerate(patch_tokens):
        batch_text = text_hypotheses[batch_index] if text_hypotheses is not None and text_hypotheses.ndim == 3 else text_hypotheses
        batch_positions = positions[batch_index] if positions is not None else None
        results.append(_foresight_single(batch_tokens, batch_text, cfg, token_norm, batch_positions))
    return _stack_results(results)


def _foresight_single(
    patch_tokens: torch.Tensor,
    text_hypotheses: torch.Tensor | None,
    cfg: ForesightConfig,
    token_norm: TokenNorm,
    positions: torch.Tensor | None,
) -> ForesightResult:
    if patch_tokens.ndim != 2:
        raise ValueError(f"Expected [N, D] patch tokens, got {tuple(patch_tokens.shape)}")
    if int(patch_tokens.shape[0]) == 0:
        raise ValueError("cannot reduce an empty visual-token sequence")

    n = int(patch_tokens.shape[0])
    d = int(patch_tokens.shape[-1])
    norm_fn = _make_norm(token_norm)
    norm_patch = norm_fn(patch_tokens).to(dtype=torch.float32)
    z_tokens = F.normalize(norm_patch, p=2, dim=-1, eps=cfg.eps)
    image_mean = norm_patch.mean(dim=0)
    z_img = F.normalize(image_mean, p=2, dim=-1, eps=cfg.eps)

    image_hypotheses, image_anchor_indices = _build_image_hypotheses(norm_patch, z_tokens, image_mean, cfg.eps)
    text_directions = _prepare_text_hypotheses(text_hypotheses, d, patch_tokens.device, cfg.eps)
    if text_directions is not None and int(text_directions.shape[0]) > 0:
        hypotheses = torch.cat([image_hypotheses, text_directions], dim=0)
    else:
        hypotheses = image_hypotheses
    hypotheses = F.normalize(hypotheses, p=2, dim=-1, eps=cfg.eps)

    m = int(hypotheses.shape[0])
    k_budget = min(int(cfg.k_max), n, m)
    k_min = min(int(cfg.k_min), k_budget)
    if k_budget <= 0:
        raise ValueError("FORESIGHT requires at least one candidate hypothesis")

    pool_indices = torch.arange(m, dtype=torch.long, device=patch_tokens.device)
    global_scores = hypotheses @ z_img
    local_scores = (z_tokens @ hypotheses.T).max(dim=0).values
    global_support = torch.softmax(global_scores, dim=0)
    local_support = torch.softmax(local_scores, dim=0)
    omega = (global_support + cfg.eps) * (local_support + cfg.eps)
    omega = omega / omega.sum().clamp_min(cfg.eps)

    candidate_order = _diversity_aware_selection(hypotheses, omega, k_budget)
    candidate_hypotheses = hypotheses[candidate_order]
    candidate_pool_indices = pool_indices[candidate_order]
    candidate_omega = omega[candidate_order]

    candidate_response = norm_patch @ candidate_hypotheses.T
    candidate_response_hat = _standardize_columns(candidate_response, cfg.eps)
    patch_affinity = torch.softmax(candidate_response_hat, dim=-1)
    patch_support = patch_affinity.mean(dim=0)
    image_confidence = candidate_omega / candidate_omega.sum().clamp_min(cfg.eps)
    evidence = (image_confidence + cfg.eps) * (patch_support + cfg.eps)
    evidence = evidence / evidence.sum().clamp_min(cfg.eps)

    score_order = torch.argsort(evidence, descending=True, stable=True)
    cumulative = torch.cumsum(evidence[score_order], dim=0)
    hits = torch.nonzero(cumulative >= float(cfg.rho), as_tuple=False).flatten()
    k_rho = int(hits[0].item()) + 1 if int(hits.numel()) else k_budget
    k_eff = max(k_min, min(k_rho, k_budget))

    active_positions = score_order[:k_eff]
    active_hypotheses = candidate_hypotheses[active_positions]
    active_indices = candidate_pool_indices[active_positions]
    q_tokens, p_hat, importance = _semantic_response(norm_patch, active_hypotheses, cfg.eps)

    seed_indices = _select_class_guided_seeds(p_hat)
    centers = q_tokens[seed_indices].clone()
    iterations = 1 + math.ceil(math.log2(max(1.0, float(n) / float(k_eff))))
    centers, assignments, iterations_used, final_change_rate = _cluster_with_repair(
        q_tokens=q_tokens,
        importance=importance,
        centers=centers,
        iterations=iterations,
        eps=cfg.eps,
    )

    result = _aggregate_prototypes(
        patch_tokens=patch_tokens,
        q_tokens=q_tokens,
        importance=importance,
        centers=centers,
        assignments=assignments,
        seed_indices=seed_indices,
        hypothesis_indices=active_indices,
        positions=positions,
        temperature=cfg.temperature,
        lambda_importance=cfg.lambda_importance,
        sort_by_position=cfg.sort_by_position,
        candidate_indices=candidate_pool_indices,
        candidate_scores=evidence,
        image_confidence=image_confidence,
        patch_support=patch_support,
        k_min=k_min,
        k_max=int(cfg.k_max),
        k_budget=k_budget,
        k_rho=k_rho,
        k_eff=k_eff,
        iterations_used=iterations_used,
        final_assignment_change_rate=final_change_rate,
        image_hypothesis_count=int(image_anchor_indices.numel()),
        text_hypothesis_count=0 if text_directions is None else int(text_directions.shape[0]),
    )
    return result


def _build_image_hypotheses(
    norm_patch: torch.Tensor,
    z_tokens: torch.Tensor,
    image_mean: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    denom = image_mean.pow(2).sum().clamp_min(eps)
    scale = (norm_patch @ image_mean) / denom
    residuals = norm_patch - scale.unsqueeze(-1) * image_mean.unsqueeze(0)
    directions = F.normalize(residuals, p=2, dim=-1, eps=eps)
    fallback = residuals.norm(dim=-1) <= eps
    if bool(fallback.any()):
        directions[fallback] = z_tokens[fallback]
    directions = F.normalize(directions, p=2, dim=-1, eps=eps)
    indices = torch.arange(int(norm_patch.shape[0]), dtype=torch.long, device=norm_patch.device)
    return directions, indices


def _prepare_text_hypotheses(
    text_hypotheses: torch.Tensor | None,
    hidden_dim: int,
    device: torch.device,
    eps: float,
) -> torch.Tensor | None:
    if text_hypotheses is None or int(text_hypotheses.numel()) == 0:
        return None
    if text_hypotheses.ndim != 2:
        raise ValueError(f"text_hypotheses must have shape [M, D], got {tuple(text_hypotheses.shape)}")
    if int(text_hypotheses.shape[-1]) != int(hidden_dim):
        raise ValueError(f"text hidden dim {int(text_hypotheses.shape[-1])} != visual hidden dim {hidden_dim}")
    text = text_hypotheses.to(device=device, dtype=torch.float32)
    text = F.layer_norm(text, text.shape[-1:])
    return F.normalize(text, p=2, dim=-1, eps=eps)


def _diversity_aware_selection(hypotheses: torch.Tensor, weights: torch.Tensor, count: int) -> torch.Tensor:
    total = int(hypotheses.shape[0])
    count = min(max(1, int(count)), total)
    selected: list[int] = []
    unavailable = torch.zeros(total, dtype=torch.bool, device=hypotheses.device)

    first = int(torch.argmax(weights).item())
    selected.append(first)
    unavailable[first] = True

    while len(selected) < count:
        selected_tensor = torch.tensor(selected, dtype=torch.long, device=hypotheses.device)
        similarities = hypotheses @ hypotheses[selected_tensor].T
        diversity = (1.0 - similarities).min(dim=-1).values.clamp_min(0.0)
        scores = weights * diversity
        scores[unavailable] = -torch.inf
        next_index = int(torch.argmax(scores).item())
        selected.append(next_index)
        unavailable[next_index] = True

    return torch.tensor(selected, dtype=torch.long, device=hypotheses.device)


def _semantic_response(
    norm_patch: torch.Tensor,
    active_hypotheses: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    responses = norm_patch @ active_hypotheses.T
    p_hat = _standardize_columns(responses, eps)
    q_tokens = F.normalize(p_hat, p=2, dim=-1, eps=eps)
    importance = p_hat.max(dim=-1).values - p_hat.mean(dim=-1)
    importance = (importance - importance.mean()) / (importance.std(unbiased=False) + eps)
    return q_tokens, p_hat, importance


def _standardize_columns(values: torch.Tensor, eps: float) -> torch.Tensor:
    mean = values.mean(dim=0, keepdim=True)
    std = values.std(dim=0, keepdim=True, unbiased=False)
    return (values - mean) / (std + eps)


def _select_class_guided_seeds(p_hat: torch.Tensor) -> torch.Tensor:
    n, k = p_hat.shape
    if k > n:
        raise ValueError("FORESIGHT requires K_eff <= number of visual tokens")
    selected: list[int] = []
    unavailable = torch.zeros(n, dtype=torch.bool, device=p_hat.device)
    for class_index in range(k):
        scores = p_hat[:, class_index].clone()
        scores[unavailable] = -torch.inf
        index = int(torch.argmax(scores).item())
        selected.append(index)
        unavailable[index] = True
    return torch.tensor(selected, dtype=torch.long, device=p_hat.device)


def _cluster_with_repair(
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    centers: torch.Tensor,
    iterations: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, int, float | None]:
    assignments = torch.zeros(int(q_tokens.shape[0]), dtype=torch.long, device=q_tokens.device)
    previous: torch.Tensor | None = None
    final_change_rate: float | None = None
    iterations_used = 0
    for _ in range(max(1, int(iterations))):
        assignments = (q_tokens @ centers.T).argmax(dim=-1)
        assignments = _repair_empty_clusters(q_tokens, centers, assignments)
        centers = _update_centers(q_tokens, importance, centers, assignments, eps)
        iterations_used += 1
        if previous is not None:
            final_change_rate = float((assignments != previous).float().mean().item())
        previous = assignments.clone()
    return centers, assignments, iterations_used, final_change_rate


def _repair_empty_clusters(q_tokens: torch.Tensor, centers: torch.Tensor, assignments: torch.Tensor) -> torch.Tensor:
    k = int(centers.shape[0])
    counts = torch.bincount(assignments, minlength=k)
    empty = torch.nonzero(counts == 0, as_tuple=False).flatten()
    if int(empty.numel()) == 0:
        return assignments
    repaired = assignments.clone()
    for empty_index in empty.tolist():
        donors = torch.nonzero(counts > 1, as_tuple=False).flatten()
        if int(donors.numel()) == 0:
            raise RuntimeError("cannot repair empty cluster without non-singleton donor")
        donor = _select_donor_cluster(q_tokens, centers, repaired, donors)
        donor_members = torch.nonzero(repaired == donor, as_tuple=False).flatten()
        donor_scores = q_tokens[donor_members] @ centers[donor]
        moved = donor_members[torch.argmin(donor_scores)]
        counts[donor] -= 1
        repaired[moved] = int(empty_index)
        counts[empty_index] += 1
    return repaired


def _select_donor_cluster(
    q_tokens: torch.Tensor,
    centers: torch.Tensor,
    assignments: torch.Tensor,
    donors: torch.Tensor,
) -> int:
    best = int(donors[0].item())
    best_dispersion = -float("inf")
    for donor in donors.tolist():
        members = torch.nonzero(assignments == int(donor), as_tuple=False).flatten()
        similarities = q_tokens[members] @ centers[int(donor)]
        dispersion = float((1.0 - similarities).mean().item())
        if dispersion > best_dispersion:
            best_dispersion = dispersion
            best = int(donor)
    return best


def _update_centers(
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    centers: torch.Tensor,
    assignments: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    updated = []
    for center_index in range(int(centers.shape[0])):
        members = torch.nonzero(assignments == center_index, as_tuple=False).flatten()
        if int(members.numel()) == 0:
            raise RuntimeError("empty cluster remained after repair")
        weights = torch.exp(importance[members]).unsqueeze(-1)
        weighted = (weights * q_tokens[members]).sum(dim=0, keepdim=True)
        updated.append(F.normalize(weighted, p=2, dim=-1, eps=eps).squeeze(0))
    return torch.stack(updated, dim=0)


def _aggregate_prototypes(
    patch_tokens: torch.Tensor,
    q_tokens: torch.Tensor,
    importance: torch.Tensor,
    centers: torch.Tensor,
    assignments: torch.Tensor,
    seed_indices: torch.Tensor,
    hypothesis_indices: torch.Tensor,
    positions: torch.Tensor | None,
    temperature: float,
    lambda_importance: float,
    sort_by_position: bool,
    candidate_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    image_confidence: torch.Tensor,
    patch_support: torch.Tensor,
    k_min: int,
    k_max: int,
    k_budget: int,
    k_rho: int,
    k_eff: int,
    iterations_used: int,
    final_assignment_change_rate: float | None,
    image_hypothesis_count: int,
    text_hypothesis_count: int,
) -> ForesightResult:
    prototypes = []
    masses = []
    soft_positions = []
    prepared_positions = _prepare_positions(positions, int(patch_tokens.shape[0]), patch_tokens.device)

    for center_index in range(int(centers.shape[0])):
        members = torch.nonzero(assignments == center_index, as_tuple=False).flatten()
        if int(members.numel()) == 0:
            raise RuntimeError("FORESIGHT cannot aggregate an empty cluster")
        semantic_scores = q_tokens[members] @ centers[center_index]
        scores = semantic_scores + float(lambda_importance) * importance[members]
        weights = torch.softmax(scores / float(temperature), dim=0).to(dtype=patch_tokens.dtype)
        prototypes.append((patch_tokens[members] * weights.unsqueeze(-1)).sum(dim=0))
        masses.append(torch.tensor(int(members.numel()), dtype=torch.long, device=patch_tokens.device))
        if prepared_positions is not None:
            pos_weights = weights.to(dtype=prepared_positions.dtype)
            soft_positions.append((prepared_positions[members] * pos_weights.unsqueeze(-1)).sum(dim=0))

    patch_result = torch.stack(prototypes, dim=0).to(dtype=patch_tokens.dtype)
    masses_result = torch.stack(masses, dim=0)
    centers_result = centers
    positions_result = torch.stack(soft_positions, dim=0) if soft_positions else None
    prototype_order = seed_indices.clone()

    if sort_by_position and positions_result is not None:
        order = _position_order(positions_result)
        patch_result = patch_result[order]
        masses_result = masses_result[order]
        centers_result = centers_result[order]
        positions_result = positions_result[order]
        hypothesis_indices = hypothesis_indices[order]
        seed_indices = seed_indices[order]
        prototype_order = prototype_order[order]
        assignments = _remap_assignments(assignments, order)

    return ForesightResult(
        patch_tokens=patch_result,
        assignments=assignments,
        centers=centers_result,
        hypothesis_indices=hypothesis_indices,
        seed_indices=seed_indices,
        masses=masses_result,
        soft_positions=positions_result,
        prototype_order=prototype_order,
        candidate_indices=candidate_indices,
        candidate_scores=candidate_scores,
        image_confidence=image_confidence,
        patch_support=patch_support,
        k_min=k_min,
        k_max=k_max,
        k_budget=k_budget,
        k_rho=k_rho,
        k_eff=k_eff,
        iterations_used=iterations_used,
        final_assignment_change_rate=final_assignment_change_rate,
        image_hypothesis_count=image_hypothesis_count,
        text_hypothesis_count=text_hypothesis_count,
    )


def _prepare_positions(positions: torch.Tensor | None, n: int, device: torch.device) -> torch.Tensor | None:
    if positions is not None:
        if tuple(positions.shape) != (n, 2):
            raise ValueError(f"positions must have shape [{n}, 2], got {tuple(positions.shape)}")
        return positions.to(device=device, dtype=torch.float32)
    side = math.isqrt(n)
    if side * side != n:
        return None
    rows = torch.arange(side, device=device, dtype=torch.float32)
    cols = torch.arange(side, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)


def _position_order(positions: torch.Tensor) -> torch.Tensor:
    width = positions[:, 1].max().clamp(min=0.0) + 1.0
    return torch.argsort(positions[:, 0] * width + positions[:, 1], stable=True)


def _remap_assignments(assignments: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(int(order.numel()), device=order.device)
    return inverse[assignments]


def _stack_results(results: list[ForesightResult]) -> ForesightResult:
    counts = {item.k_eff for item in results}
    if len(counts) != 1:
        raise ValueError("FORESIGHT produced different K_eff values inside one batch; run samples one by one.")
    first = results[0]
    return ForesightResult(
        patch_tokens=torch.stack([item.patch_tokens for item in results], dim=0),
        assignments=torch.stack([item.assignments for item in results], dim=0),
        centers=torch.stack([item.centers for item in results], dim=0),
        hypothesis_indices=torch.stack([item.hypothesis_indices for item in results], dim=0),
        seed_indices=torch.stack([item.seed_indices for item in results], dim=0),
        masses=torch.stack([item.masses for item in results], dim=0),
        soft_positions=torch.stack([item.soft_positions for item in results], dim=0)
        if all(item.soft_positions is not None for item in results)
        else None,
        prototype_order=torch.stack([item.prototype_order for item in results], dim=0),
        candidate_indices=torch.stack([item.candidate_indices for item in results], dim=0),
        candidate_scores=torch.stack([item.candidate_scores for item in results], dim=0),
        image_confidence=torch.stack([item.image_confidence for item in results], dim=0),
        patch_support=torch.stack([item.patch_support for item in results], dim=0),
        k_min=first.k_min,
        k_max=first.k_max,
        k_budget=first.k_budget,
        k_rho=first.k_rho,
        k_eff=first.k_eff,
        iterations_used=first.iterations_used,
        final_assignment_change_rate=first.final_assignment_change_rate,
        image_hypothesis_count=first.image_hypothesis_count,
        text_hypothesis_count=first.text_hypothesis_count,
    )


def _make_norm(token_norm: TokenNorm) -> Callable[[torch.Tensor], torch.Tensor]:
    if token_norm is None:
        return lambda values: F.layer_norm(values.float(), values.shape[-1:])
    return token_norm


def _validate_config(cfg: ForesightConfig) -> None:
    if cfg.k_min <= 0 or cfg.k_max <= 0:
        raise ValueError("k_min and k_max must be positive")
    if cfg.k_min > cfg.k_max:
        raise ValueError("k_min must be <= k_max")
    if cfg.k_text < 0:
        raise ValueError("k_text must be non-negative")
    if not 0.0 < cfg.rho <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    if cfg.eps <= 0:
        raise ValueError("eps must be positive")
    if cfg.temperature <= 0:
        raise ValueError("temperature must be positive")
