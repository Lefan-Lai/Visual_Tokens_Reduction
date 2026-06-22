from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ProtoReduceResult:
    tokens: torch.Tensor
    assignments: torch.Tensor
    centers: torch.Tensor


def proto_reduce(
    tokens: torch.Tensor,
    num_prototypes: int,
    iterations: int = 5,
    temperature: float = 0.07,
    eps: float = 1e-6,
    init: str = "farthest",
) -> torch.Tensor:
    """Reduce patch tokens to prototype tokens with spherical clustering.

    Args:
        tokens: Tensor with shape ``[n, d]`` or ``[batch, n, d]``.
        num_prototypes: Target number of prototype tokens.
        iterations: Number of hard spherical clustering iterations.
        temperature: Softmax temperature for the final per-cluster pooling.
        eps: Numerical stability value used for normalization.
        init: Center initialization method. Currently ``farthest`` or
            ``kmeans++``.

    Returns:
        Reduced tokens with shape ``[m, d]`` or ``[batch, m, d]`` where
        ``m = min(num_prototypes, n)``.
    """

    return proto_reduce_with_info(
        tokens=tokens,
        num_prototypes=num_prototypes,
        iterations=iterations,
        temperature=temperature,
        eps=eps,
        init=init,
    ).tokens


def proto_reduce_with_info(
    tokens: torch.Tensor,
    num_prototypes: int,
    iterations: int = 5,
    temperature: float = 0.07,
    eps: float = 1e-6,
    init: str = "farthest",
) -> ProtoReduceResult:
    if tokens.ndim == 2:
        return _proto_reduce_single(tokens, num_prototypes, iterations, temperature, eps, init)
    if tokens.ndim != 3:
        raise ValueError(f"Expected [n, d] or [batch, n, d] tokens, got shape {tuple(tokens.shape)}")

    reduced: list[torch.Tensor] = []
    assignments: list[torch.Tensor] = []
    centers: list[torch.Tensor] = []
    for batch_tokens in tokens:
        result = _proto_reduce_single(batch_tokens, num_prototypes, iterations, temperature, eps, init)
        reduced.append(result.tokens)
        assignments.append(result.assignments)
        centers.append(result.centers)
    return ProtoReduceResult(
        tokens=torch.stack(reduced, dim=0),
        assignments=torch.stack(assignments, dim=0),
        centers=torch.stack(centers, dim=0),
    )


def _proto_reduce_single(
    tokens: torch.Tensor,
    num_prototypes: int,
    iterations: int,
    temperature: float,
    eps: float,
    init: str,
) -> ProtoReduceResult:
    if tokens.ndim != 2:
        raise ValueError(f"Expected [n, d] tokens, got shape {tuple(tokens.shape)}")
    if num_prototypes <= 0:
        raise ValueError("num_prototypes must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    n_tokens = int(tokens.shape[0])
    if n_tokens == 0:
        raise ValueError("cannot reduce an empty token sequence")

    target = min(int(num_prototypes), n_tokens)
    if target == n_tokens:
        normalized = _normalize(tokens.float(), eps)
        assignments = torch.arange(n_tokens, device=tokens.device)
        return ProtoReduceResult(tokens=tokens, assignments=assignments, centers=normalized)

    work = tokens.float()
    normalized = _normalize(work, eps)
    centers = _initialize_centers(normalized, target, init, eps)

    assignments = torch.zeros(n_tokens, dtype=torch.long, device=tokens.device)
    for _ in range(max(0, int(iterations))):
        similarity = normalized @ centers.T
        assignments = similarity.argmax(dim=-1)
        centers = _update_centers(normalized, centers, assignments, eps)

    similarity = normalized @ centers.T
    assignments = similarity.argmax(dim=-1)
    reduced = []
    for index in range(target):
        member_mask = assignments == index
        if not bool(member_mask.any()):
            fallback_index = _least_represented_token(normalized, centers)
            reduced.append(tokens[fallback_index])
            continue
        member_tokens = tokens[member_mask]
        member_normalized = normalized[member_mask]
        scores = member_normalized @ centers[index]
        weights = torch.softmax(scores / float(temperature), dim=0).to(tokens.dtype)
        reduced.append((member_tokens * weights.unsqueeze(-1)).sum(dim=0))

    return ProtoReduceResult(
        tokens=torch.stack(reduced, dim=0).to(tokens.dtype),
        assignments=assignments,
        centers=centers,
    )


def _normalize(values: torch.Tensor, eps: float) -> torch.Tensor:
    return F.normalize(values, p=2, dim=-1, eps=eps)


def _initialize_centers(tokens: torch.Tensor, num_centers: int, init: str, eps: float) -> torch.Tensor:
    if init not in {"farthest", "kmeans++"}:
        raise ValueError(f"Unsupported init: {init}")

    if init == "kmeans++":
        return _kmeans_plus_plus(tokens, num_centers, eps)
    return _farthest_first(tokens, num_centers, eps)


def _farthest_first(tokens: torch.Tensor, num_centers: int, eps: float) -> torch.Tensor:
    mean = _normalize(tokens.mean(dim=0, keepdim=True), eps).squeeze(0)
    first = torch.argmax(tokens @ mean)
    selected = [int(first.item())]

    min_distance = 1.0 - (tokens @ tokens[first])
    for _ in range(1, num_centers):
        candidate = torch.argmax(min_distance)
        selected.append(int(candidate.item()))
        candidate_distance = 1.0 - (tokens @ tokens[candidate])
        min_distance = torch.minimum(min_distance, candidate_distance)

    return tokens[selected].clone()


def _kmeans_plus_plus(tokens: torch.Tensor, num_centers: int, eps: float) -> torch.Tensor:
    del eps
    selected = [0]
    min_distance = 1.0 - (tokens @ tokens[0])
    for _ in range(1, num_centers):
        weights = torch.clamp(min_distance, min=0.0)
        total = weights.sum()
        if float(total.item()) <= 0:
            candidate = torch.argmax(min_distance)
        else:
            candidate = torch.multinomial(weights / total, num_samples=1).squeeze(0)
        selected.append(int(candidate.item()))
        candidate_distance = 1.0 - (tokens @ tokens[candidate])
        min_distance = torch.minimum(min_distance, candidate_distance)
    return tokens[selected].clone()


def _update_centers(
    tokens: torch.Tensor,
    centers: torch.Tensor,
    assignments: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    updated = []
    for index in range(int(centers.shape[0])):
        mask = assignments == index
        if bool(mask.any()):
            updated.append(_normalize(tokens[mask].mean(dim=0, keepdim=True), eps).squeeze(0))
            continue
        updated.append(tokens[_least_represented_token(tokens, centers)])
    return torch.stack(updated, dim=0)


def _least_represented_token(tokens: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    best_similarity = (tokens @ centers.T).max(dim=-1).values
    return best_similarity.argmin()
