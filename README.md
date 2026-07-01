# FORESIGHT for LLaVA-1.5-7B on POPE

This repository contains only the FORESIGHT algorithm version used for the
LLaVA-1.5-7B POPE run whose metadata says:

- `reduction_strategy = foresight_dynamic_hypothesis_reduction`
- `hypothesis_generation = per_image_token_residual_plus_text_embeddings`
- `candidate_selection = omega_weighted_diversity_selection`
- `evidence_rule = multiplicative_image_confidence_and_token_support`

This is not the later protected-token / dual-space-clustering variant.

## Files

- `FORESIGHT_ALGORITHM.md`: full algorithm description.
- `IMPLEMENTATION_DETAILS.md`: hyperparameters, implementation mapping, and logging fields.
- `foresight.py`: training-free visual-token reduction implementation.
- `run_llava7b_pope.py`: LLaVA-1.5-7B POPE evaluation runner using FORESIGHT.
- `launch_llava7b_pope.sh`: server launch script.

## Run

```bash
python run_llava7b_pope.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --dataset-name lmms-lab/POPE \
  --split test \
  --category all \
  --k-min 32 \
  --k-max 128 \
  --k-text 64 \
  --rho 0.90 \
  --load-in-4bit \
  --output-dir results/llava7b_pope_foresight
```

On the Auckland server, use:

```bash
./launch_llava7b_pope.sh
```

The runner prints per-sample progress and writes JSONL/CSV/summary files.
