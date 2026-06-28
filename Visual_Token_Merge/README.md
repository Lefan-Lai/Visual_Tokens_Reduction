# Early-SemReduce, Flood-SemReduce, Fast Flood-SemReduce, K-SemReduce, and Answer-Adaptive K-SemReduce Visual Token Merge

This folder contains a standalone PyTorch implementation of **Early-SemReduce:
Training-Free Semantic Response Guided Visual Token Reduction**.

The method inserts token reduction after an early/intermediate vision-transformer
layer. It runs the first `ell` visual blocks at full resolution, clusters patch
tokens in frozen classifier-response space, merges each semantic cluster back in
the original visual-token space, and sends only `m` prototype tokens through the
remaining visual blocks.

It also contains **Flood-SemReduce: Training-Free Semantic-Admissible Region
Growing for Visual Token Reduction**. Flood-SemReduce removes fixed-budget
K-means clustering. It initializes every patch as a region on the 2D token grid,
then repeatedly merges adjacent regions only when their semantic responses are
compatible and their importance boundary is admissible. Its output token count
`M` is dynamic and image-dependent.

The latest implementation also contains **Fast Flood-SemReduce**. This is the
faster connected-component version of the same idea: it builds all 4-neighbor
patch edges once, keeps only semantic-admissible edges, runs connected
components, optionally repairs inconsistent regions once, and aggregates each
final region into one visual token. It does not use a priority queue and it does
not update region-pair scores after every merge.

It also contains **K-SemReduce: Class-Prototype Guided Training-Free Visual
Token Reduction**. K-SemReduce removes the independent prototype count `m`,
protected-anchor count `b`, and Top-B anchor path. Its single core control
variable is `K`: the Top-K candidate semantic class count, semantic response
dimension, cluster count, and output prototype-token count are all the same.

The newest variant is **Answer-Adaptive K-SemReduce**. Instead of using a fixed
K, it builds a question-related semantic hypothesis set `H_Q` from the current
question, relation words, quoted answer concepts, colors, and category hints.
The requested reduced token count is `K_Q = |H_Q| * hypothesis_multiplier`,
clamped only when it exceeds the number of available patch tokens.

## Files

- `early_semreduce/reducer.py`: core Early-SemReduce implementation.
- `early_semreduce/flood_reducer.py`: core Flood-SemReduce region-growing
  implementation.
- `early_semreduce/fast_flood_reducer.py`: Fast Flood-SemReduce
  connected-component implementation.
- `early_semreduce/k_reducer.py`: K-SemReduce class-prototype guided
  implementation.
- `early_semreduce/answer_adaptive_reducer.py`: Answer-Adaptive K-SemReduce
  dynamic semantic-hypothesis implementation.
- `early_semreduce/vit_wrapper.py`: generic timm-style ViT forward helper.
- `run_llava13b_mme.py`: MME yes/no evaluation runner with overall,
  dimension-level, and category-level metrics.
- `scripts/demo_semreduce.py`: synthetic smoke demo.
- `tests/test_early_semreduce.py`: unit tests for shape, assignment, sorting,
  anchors, and no-reduction behavior.
- `tests/test_flood_semreduce.py`: unit tests for dynamic region counts and
  CLS preservation.
- `tests/test_fast_flood_semreduce.py`: unit tests for the fast
  connected-component reducer.
- `tests/test_k_semreduce.py`: unit tests for exact-K output, K clamping, and
  non-duplicate class-guided seed selection.
- `tests/test_answer_adaptive_k_semreduce.py`: unit tests for dynamic
  hypothesis-count output and K_Q clamping.

## Install

```bash
cd /home/llai933/SEER/Visual_Token_Merge
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the local Windows checkout:

```powershell
cd "D:\Codex - File\AAAI-SEER\Visual_Token_Merge"
python -m pip install -r requirements.txt
```

## Quick Demo

```bash
python scripts/demo_semreduce.py \
  --patches 196 \
  --dim 768 \
  --classes 1000 \
  --prototype-tokens 64 \
  --candidate-classes 64 \
  --anchors 8
```

Expected output is a JSON shape summary. For example, `197` input tokens become
`65` output tokens: one CLS token plus `64` semantic prototype tokens.

## Core API

```python
import torch
from early_semreduce import EarlySemReduce, SemReduceConfig

batch = 2
patches = 196
dim = 768
classes = 1000

sequence = torch.randn(batch, patches + 1, dim)  # [CLS, patch_1, ..., patch_n]
classifier_weight = torch.randn(classes, dim)    # frozen classifier head

reducer = EarlySemReduce(
    SemReduceConfig(
        num_prototypes=64,
        candidate_classes=64,
        num_anchors=8,
        iterations=5,
        temperature=0.07,
        lambda_importance=0.25,
        lambda_diversity=1.0,
        gamma=1.0,
    )
)

result = reducer(sequence, classifier_weight)
reduced_sequence = result.sequence
cluster_assignments = result.assignments
prototype_masses = result.masses
```

Flood-SemReduce uses a separate config because it does not take
`num_prototypes`:

```python
from early_semreduce import FloodSemReduce, FloodSemReduceConfig

reducer = FloodSemReduce(FloodSemReduceConfig(candidate_classes=64))
result = reducer(sequence, classifier_weight)

reduced_sequence = result.sequence       # [B, 1 + M, D]
dynamic_tokens = result.patch_tokens.shape[-2]
region_masses = result.masses
```

Fast Flood-SemReduce exposes the same dynamic-token interface:

```python
from early_semreduce import FastFloodSemReduce, FastFloodSemReduceConfig

reducer = FastFloodSemReduce(FastFloodSemReduceConfig(candidate_classes=64))
result = reducer(sequence, classifier_weight)

reduced_sequence = result.sequence       # [B, 1 + M, D]
dynamic_tokens = result.patch_tokens.shape[-2]
region_masses = result.masses
```

K-SemReduce uses `K` as both the candidate-class count and output prototype
count:

```python
from early_semreduce import KSemReduce, KSemReduceConfig

reducer = KSemReduce(
    KSemReduceConfig(
        num_semantic_classes=64,
        iterations=3,
        temperature=0.1,
        lambda_importance=0.25,
        gamma=1.0,
    )
)
result = reducer(sequence, classifier_weight)

reduced_sequence = result.sequence       # [B, 1 + K, D]
prototype_tokens = result.patch_tokens
```

Answer-Adaptive K-SemReduce takes a dynamic semantic hypothesis matrix:

```python
from early_semreduce import AnswerAdaptiveKSemReduceConfig, answer_adaptive_k_semreduce

result = answer_adaptive_k_semreduce(
    patch_tokens=patch_tokens,
    hypothesis_embeddings=W_H,  # [K_Q, D]
    config=AnswerAdaptiveKSemReduceConfig(
        iterations=3,
        temperature=0.1,
        lambda_importance=0.25,
        gamma=1.0,
    ),
)
```

For LLaVA, `run_llava13b_pope.py` supports:

```bash
PYTHONPATH=. python run_llava13b_pope.py \
  --model-id llava-hf/llava-1.5-13b-hf \
  --methods vanilla,k_semreduce \
  --limit 100 \
  --category adversarial \
  --candidate-classes 64 \
  --k-cluster-iters 3 \
  --k-temperature 0.1 \
  --load-in-4bit \
  --output-dir ~/results/llava13b_k_semreduce_pope_adversarial_100
```

For MME, `run_llava13b_mme.py` supports 300-image evaluation and writes
`summary.json`, `category_metrics.csv`, and `dimension_metrics.csv`:

```bash
PYTHONPATH=. python run_llava13b_mme.py \
  --model-id llava-hf/llava-1.5-13b-hf \
  --methods vanilla,answer_adaptive_k_semreduce \
  --limit-images 300 \
  --sampling stratified \
  --adaptive-cluster-iters 3 \
  --adaptive-temperature 0.1 \
  --adaptive-lambda-importance 0.25 \
  --adaptive-gamma 1.0 \
  --adaptive-hypothesis-multiplier 1 \
  --load-in-4bit \
  --output-dir ~/results/llava13b_answer_adaptive_k_semreduce_mme300
```

The LLaVA runner first precomputes reduced image features to discover the
dynamic output count `M`, then rewrites the number of `<image>` placeholders to
match `M` before calling the language model.

By default, the reducer applies parameter-free layer norm before projecting
tokens onto the classifier head. To use the frozen norm from a real model:

```python
reducer = EarlySemReduce(config, token_norm=model.norm)
```

## timm-Style ViT Integration

```python
from early_semreduce import EarlySemReduce, SemReduceConfig, forward_timm_vit_with_semreduce

config = SemReduceConfig(num_prototypes=64, candidate_classes=64, num_anchors=8)
reducer = EarlySemReduce(config, token_norm=model.norm)

logits = forward_timm_vit_with_semreduce(
    model=model,
    images=images,
    reducer=reducer,
    reduction_layer=6,
)
```

The helper uses `model.head.weight` as the frozen classifier head unless another
classifier tensor/module is passed explicitly.

## Algorithm Summary

For intermediate tokens

```text
X^(ell) = [x_cls^(ell), x_1^(ell), ..., x_n^(ell)]
```

Early-SemReduce:

1. Selects candidate classes from `W_cls LN(x_cls)`.
2. Computes patch semantic responses `p_i = W_S LN(x_i)`.
3. Standardizes responses across patches and L2-normalizes them into `q_i`.
4. Protects top-importance patches as singleton semantic anchors.
5. Clusters remaining patches in semantic-response space.
6. Aggregates original visual tokens with importance-aware soft weights.
7. Optionally sorts prototypes by soft 2D position.
8. Continues the remaining Transformer blocks on `[CLS, r_1, ..., r_m]`.

The implementation is training-free: it adds no learnable parameters and uses
only frozen model weights plus deterministic tensor operations.

## Flood-SemReduce Summary

Flood-SemReduce keeps the semantic-response projection from Early-SemReduce but
replaces fixed-budget clustering with region growing:

1. Compute top-`k` candidate semantic rows from a frozen head. The default used
   in the LLaVA POPE run is `k = 64`.
2. Compute patch-level semantic responses and normalize them into `q_i`.
3. Compute semantic importance for every patch.
4. Build a 4-neighbor grid over patch positions.
5. Use Otsu thresholds from the current image's adjacent-patch similarity and
   importance-difference distributions.
6. Initialize every patch as one region.
7. Push admissible adjacent region pairs into a priority queue.
8. Repeatedly merge the strongest admissible region pair if it passes a
   post-merge consistency check.
9. Stop when no adjacent regions can legally merge.
10. Aggregate each final region back in the original visual-token space.

The final token count is not fixed. In the 100-sample POPE adversarial run,
Flood-SemReduce reduced LLaVA image features from 576 tokens to an average of
about 201 tokens, with a range of 91 to 289 tokens.

## Fast Flood-SemReduce Summary

Fast Flood-SemReduce keeps the same semantic-response projection, but replaces
priority-queue growing with one graph construction pass:

1. Compute top-`k` candidate semantic rows from a frozen head. The default used
   in the LLaVA POPE command is `k = 64`.
2. Compute patch-level semantic responses and L2-normalize them into `q_i`.
3. Compute normalized patch importance scores.
4. Build the 4-neighbor image grid edges once.
5. Compute adjacent-patch semantic similarities and importance differences.
6. Use Otsu thresholds from the current image to get a semantic threshold and an
   importance-difference threshold.
7. Keep an edge only when the two patches are semantically similar enough and
   their importance difference is small enough.
8. Run connected components over the kept edges.
9. For each component, run one region consistency check against its semantic
   center and mean importance.
10. If a component is inconsistent, remove its bad internal links once and split
    it by connected components again.
11. Aggregate each final region in the original visual-token space with
    semantic-center similarity plus patch importance.
12. Sort output tokens by their soft 2D positions so downstream order stays
    image-like.

The output token count is still dynamic, but the reducer avoids repeated
merge-score recomputation. In code, this is exposed as `fast_flood_semreduce`.

## K-SemReduce Summary

K-SemReduce keeps the semantic-response projection but simplifies the control
surface. There is no independent `m`, no protected-anchor count `b`, and no
Top-B singleton anchor path.

1. Use the CLS token and frozen classifier head to select Top-K candidate
   semantic classes.
2. Project each patch token onto those K classifier rows to get a `[N, K]`
   semantic response matrix.
3. Standardize every candidate-class response dimension across patches.
4. L2-normalize each patch response vector into `q_i`.
5. Compute semantic importance from standardized responses.
6. For each candidate class, select the strongest not-yet-used patch as a
   class-guided seed.
7. Initialize exactly K semantic centers from those K distinct seeds.
8. Assign all patches to their closest semantic center.
9. Repair empty clusters by moving the least-fitting patch from a non-singleton
   donor cluster.
10. Update centers with importance-aware weighted means for `T` iterations.
11. Aggregate every cluster back in the original visual hidden space with
    `score_i = dot(q_i, mu_j) + lambda_imp * u_i`.
12. Sort prototype tokens by soft 2D position.

The recommended starting hyperparameters are `K = 64`, `T = 3`, `tau = 0.1`,
`lambda_imp = 0.25`, and `gamma = 1.0`.

## Answer-Adaptive K-SemReduce Summary

Answer-Adaptive K-SemReduce replaces fixed `K` with a per-question semantic
hypothesis set:

1. Extract question visual concepts, relation words, quoted answer concepts, and
   category hints.
2. Expand those hypotheses by `hypothesis_multiplier`. For multiplier 1, the
   set is exactly `H_Q`; for multiplier 2 or 3, each hypothesis receives
   additional visual-evidence/detail variants.
3. Convert every expanded hypothesis string into a frozen semantic vector using the
   model's token embedding space.
4. Use the resulting matrix `W_H` as the semantic head.
5. Run K-SemReduce with `K_Q = |H_Q| * hypothesis_multiplier`.

For MME yes/no questions, the implementation does not use `yes` or `no` as
visual hypotheses. It instead builds hypotheses such as object words, colors,
relation concepts, code/text/category hints, and quoted answer concepts.

## Test

```bash
python -m pytest tests
```
