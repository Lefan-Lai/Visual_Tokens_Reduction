from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from early_semreduce import FloodSemReduce, FloodSemReduceConfig, flood_semreduce  # noqa: E402


def test_flood_semreduce_patch_only_dynamic_regions() -> None:
    torch.manual_seed(23)
    patch_tokens = torch.randn(16, 12)
    classifier = torch.randn(8, 12)

    result = flood_semreduce(
        patch_tokens=patch_tokens,
        classifier=classifier,
        candidate_classes=6,
    )

    assert result.sequence is None
    assert result.patch_tokens.shape[1] == 12
    assert 1 <= result.patch_tokens.shape[0] <= 16
    assert torch.equal(result.masses.sum(), torch.tensor(16))
    assert result.assignments.shape == (16,)
    assert int(result.assignments.min()) >= 0
    assert int(result.assignments.max()) < result.patch_tokens.shape[0]
    assert result.selected_classes.shape == (6,)


def test_flood_semreduce_sequence_keeps_cls() -> None:
    torch.manual_seed(29)
    sequence = torch.randn(1, 10, 16)
    classifier = torch.randn(12, 16)
    reducer = FloodSemReduce(FloodSemReduceConfig(candidate_classes=5))

    result = reducer(sequence, classifier)

    assert result.sequence is not None
    assert result.sequence.shape[0] == 1
    assert result.sequence.shape[-1] == 16
    assert torch.equal(result.sequence[:, 0], sequence[:, 0])
    assert torch.equal(result.masses.sum(dim=-1), torch.tensor([9]))
