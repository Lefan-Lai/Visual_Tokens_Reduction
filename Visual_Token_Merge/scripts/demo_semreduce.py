from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from early_semreduce import EarlySemReduce, SemReduceConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic Early-SemReduce demo.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patches", type=int, default=196)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--prototype-tokens", type=int, default=64)
    parser.add_argument("--candidate-classes", type=int, default=64)
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    sequence = torch.randn(args.batch_size, args.patches + 1, args.dim)
    classifier = torch.randn(args.classes, args.dim)
    reducer = EarlySemReduce(
        SemReduceConfig(
            num_prototypes=args.prototype_tokens,
            candidate_classes=args.candidate_classes,
            num_anchors=args.anchors,
            iterations=args.iterations,
        )
    )
    result = reducer(sequence, classifier)
    summary = {
        "input_sequence_shape": list(sequence.shape),
        "reduced_sequence_shape": list(result.sequence.shape),
        "selected_classes_shape": list(result.selected_classes.shape),
        "assignments_shape": list(result.assignments.shape),
        "mass_per_sample": result.masses.sum(dim=-1).tolist(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
