from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from early_semreduce import KSemReduce, KSemReduceConfig, k_semreduce  # noqa: E402
from early_semreduce.k_reducer import _select_class_guided_seeds  # noqa: E402


def test_k_semreduce_patch_only_outputs_exact_k() -> None:
    torch.manual_seed(41)
    patch_tokens = torch.randn(25, 16)
    classifier = torch.randn(20, 16)

    result = k_semreduce(
        patch_tokens=patch_tokens,
        classifier=classifier,
        num_semantic_classes=7,
        iterations=2,
    )

    assert result.sequence is None
    assert result.patch_tokens.shape == (7, 16)
    assert result.centers.shape == (7, 7)
    assert result.selected_classes.shape == (7,)
    assert result.anchors.numel() == 0
    assert torch.equal(result.masses.sum(), torch.tensor(25))
    assert torch.all(result.masses > 0)
    assert result.assignments.shape == (25,)
    assert int(result.assignments.min()) >= 0
    assert int(result.assignments.max()) < 7


def test_k_semreduce_sequence_keeps_cls_and_clamps_k_to_patch_count() -> None:
    torch.manual_seed(43)
    sequence = torch.randn(1, 6, 12)
    classifier = torch.randn(30, 12)
    reducer = KSemReduce(KSemReduceConfig(num_semantic_classes=10, iterations=1))

    result = reducer(sequence, classifier)

    assert result.sequence is not None
    assert result.sequence.shape == (1, 6, 12)
    assert result.patch_tokens.shape == (1, 5, 12)
    assert torch.equal(result.sequence[:, 0], sequence[:, 0])
    assert torch.equal(result.masses.sum(dim=-1), torch.tensor([5]))
    assert torch.all(result.masses > 0)


def test_k_semreduce_non_duplicate_seed_selection() -> None:
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
