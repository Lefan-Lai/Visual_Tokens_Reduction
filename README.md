# Visual Tokens Reduction

This repository now contains a clean implementation of **K-SemReduce:
Class-Prototype Guided Training-Free Visual Token Reduction**.

The current version removes the old independent controls:

```text
no m
no b
no TopB
no protected anchors
```

The only core control variable is `K`:

```text
K = number of Top-K candidate semantic classes
K = semantic response dimension
K = number of clustering centers
K = number of output prototype tokens
```

## Files

- `k_semreduce.py`: standalone PyTorch implementation of K-SemReduce.
- `run_llava13b_mme.py`: LLaVA-1.5-13B MME runner comparing `vanilla` and
  `k_semreduce`.
- `K_SEMREDUCE_ALGORITHM.md`: detailed algorithm explanation.
- `tests/test_k_semreduce.py`: unit tests for exact-K output, K clamping, and
  non-duplicate class-guided seeds.
- `requirements.txt`: Python dependencies.

## MME Run

```bash
PYTHONPATH=. python run_llava13b_mme.py \
  --model-id llava-hf/llava-1.5-13b-hf \
  --methods vanilla,k_semreduce \
  --limit-images 300 \
  --sampling stratified \
  --candidate-classes 64 \
  --cluster-iters 3 \
  --temperature 0.1 \
  --lambda-importance 0.25 \
  --gamma 1.0 \
  --load-in-4bit \
  --output-dir ~/results/llava13b_k_semreduce_mme300_k64
```

For a quick smoke run:

```bash
PYTHONPATH=. python run_llava13b_mme.py \
  --methods vanilla,k_semreduce \
  --limit-images 5 \
  --candidate-classes 64 \
  --load-in-4bit \
  --output-dir ~/results/llava13b_k_semreduce_mme5_k64
```

The runner writes:

```text
vanilla.jsonl
k_semreduce.jsonl
summary.json
category_metrics.csv
dimension_metrics.csv
```

## Implementation Note for LLaVA

The algorithm description is written for inserting reduction after an
intermediate visual encoder layer `ell`. In this LLaVA-1.5-13B runner, the
practical insertion point is after the vision tower and multimodal projector,
because the Hugging Face LLaVA interface exposes image features there. The
runner then rewrites the number of `<image>` placeholders to match the reduced
feature count.

LLaVA does not expose an ImageNet-style visual classifier head at this stage, so
the runner uses the frozen language embedding or LM head whose hidden dimension
matches the image features as the surrogate semantic head. The K-SemReduce
algorithm itself is unchanged: it still selects Top-K classifier rows, uses
those rows to define the semantic response space, initializes one non-duplicate
seed per candidate class, clusters all patches, and outputs exactly K prototype
tokens unless K is larger than the number of input patch tokens.

## Test

```bash
python -m pytest tests
```

