from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from early_semreduce import (  # noqa: E402
    AnswerAdaptiveKSemReduce,
    AnswerAdaptiveKSemReduceConfig,
    answer_adaptive_k_semreduce,
)
from run_llava13b_mme import expand_hypotheses  # noqa: E402


def test_answer_adaptive_k_semreduce_uses_hypothesis_count() -> None:
    torch.manual_seed(47)
    patch_tokens = torch.randn(20, 12)
    hypothesis_embeddings = torch.randn(6, 12)

    result = answer_adaptive_k_semreduce(
        patch_tokens=patch_tokens,
        hypothesis_embeddings=hypothesis_embeddings,
        iterations=2,
    )

    assert result.sequence is None
    assert result.patch_tokens.shape == (6, 12)
    assert result.centers.shape == (6, 6)
    assert result.selected_classes.shape == (6,)
    assert torch.equal(result.masses.sum(), torch.tensor(20))
    assert torch.all(result.masses > 0)


def test_answer_adaptive_k_semreduce_clamps_to_patch_count() -> None:
    torch.manual_seed(53)
    sequence = torch.randn(1, 6, 10)
    hypothesis_embeddings = torch.randn(12, 10)
    reducer = AnswerAdaptiveKSemReduce(AnswerAdaptiveKSemReduceConfig(iterations=1))

    result = reducer(sequence, hypothesis_embeddings)

    assert result.sequence is not None
    assert result.sequence.shape == (1, 6, 10)
    assert result.patch_tokens.shape == (1, 5, 10)
    assert torch.equal(result.sequence[:, 0], sequence[:, 0])
    assert torch.equal(result.masses.sum(dim=-1), torch.tensor([5]))


def test_expand_hypotheses_uses_multiplier_without_min_max_bounds() -> None:
    base = ["cat", "cat", "red object"]

    expanded = expand_hypotheses(base, multiplier=3)

    assert expanded == [
        "cat",
        "red object",
        "visual evidence of cat",
        "visual evidence of red object",
        "detailed cat region",
        "detailed red object region",
    ]
