from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from k_semreduce import KSemReduce, KSemReduceConfig, k_semreduce  # noqa: E402
from k_semreduce import _select_class_guided_seeds  # noqa: E402


def test_k_semreduce_outputs_exact_k() -> None:
    torch.manual_seed(41)
    patch_tokens = torch.randn(25, 16)
    classifier = torch.randn(20, 16)

    result = k_semreduce(
        patch_tokens=patch_tokens,
        classifier=classifier,
        num_semantic_classes=7,
        iterations=2,
    )

    assert result.patch_tokens.shape == (7, 16)
    assert result.centers.shape == (7, 7)
    assert result.selected_classes.shape == (7,)
    assert result.masses.sum().item() == 25
    assert torch.all(result.masses > 0)
    assert result.assignments.shape == (25,)
    assert int(result.assignments.min()) >= 0
    assert int(result.assignments.max()) < 7
    assert result.requested_k == 7
    assert result.actual_k == 7


def test_k_semreduce_clamps_k_to_patch_count() -> None:
    torch.manual_seed(43)
    patch_tokens = torch.randn(1, 5, 12)
    classifier = torch.randn(30, 12)
    reducer = KSemReduce(KSemReduceConfig(num_semantic_classes=10, iterations=1))

    result = reducer(patch_tokens, classifier)

    assert result.patch_tokens.shape == (1, 5, 12)
    assert result.centers.shape == (1, 5, 5)
    assert result.masses.sum(dim=-1).tolist() == [5]
    assert torch.all(result.masses > 0)
    assert result.requested_k == 10
    assert result.actual_k == 5


def test_class_guided_seed_selection_is_non_duplicate() -> None:
    p_hat = torch.tensor(
        [
            [5.0, 4.9, 4.8],
            [4.0, 3.0, 2.0],
            [3.0, 4.8, 4.7],
            [2.0, 1.0, 4.6],
        ]
    )

    seeds = _select_class_guided_seeds(p_hat)

    assert seeds.tolist() == [0, 2, 3]

