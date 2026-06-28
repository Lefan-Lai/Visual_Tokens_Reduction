from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from early_semreduce import EarlySemReduce, SemReduceConfig, early_semreduce  # noqa: E402


def test_reduce_sequence_shape_and_assignments() -> None:
    torch.manual_seed(7)
    sequence = torch.randn(2, 65, 32)
    classifier = torch.randn(20, 32)
    reducer = EarlySemReduce(
        SemReduceConfig(
            num_prototypes=16,
            candidate_classes=12,
            num_anchors=4,
            iterations=3,
        )
    )

    result = reducer(sequence, classifier)

    assert result.sequence is not None
    assert result.sequence.shape == (2, 17, 32)
    assert result.patch_tokens.shape == (2, 16, 32)
    assert result.assignments.shape == (2, 64)
    assert result.selected_classes.shape == (2, 12)
    assert torch.equal(result.sequence[:, 0], sequence[:, 0])
    assert torch.equal(result.masses.sum(dim=-1), torch.tensor([64, 64]))
    assert int(result.assignments.min()) >= 0
    assert int(result.assignments.max()) < 16


def test_patch_only_function_supports_positions() -> None:
    torch.manual_seed(11)
    patch_tokens = torch.randn(49, 24)
    cls_token = torch.randn(24)
    classifier = torch.randn(10, 24)
    y, x = torch.meshgrid(torch.arange(7), torch.arange(7), indexing="ij")
    positions = torch.stack([y.flatten(), x.flatten()], dim=-1).float()

    result = early_semreduce(
        patch_tokens,
        classifier,
        cls_token=cls_token,
        positions=positions,
        num_prototypes=9,
        candidate_classes=6,
        num_anchors=2,
        iterations=2,
    )

    assert result.sequence is None
    assert result.patch_tokens.shape == (9, 24)
    assert result.soft_positions is not None
    keys = result.soft_positions[:, 0] * 8 + result.soft_positions[:, 1]
    assert torch.all(keys[:-1] <= keys[1:])
    assert torch.equal(result.masses.sum(), torch.tensor(49))


def test_no_reduction_returns_original_tokens() -> None:
    torch.manual_seed(13)
    sequence = torch.randn(1, 5, 8)
    classifier = torch.randn(4, 8)
    reducer = EarlySemReduce(SemReduceConfig(num_prototypes=4, candidate_classes=None))

    result = reducer(sequence, classifier)

    assert result.sequence is not None
    assert torch.equal(result.sequence, sequence)
    assert torch.equal(result.assignments, torch.arange(4).unsqueeze(0))
