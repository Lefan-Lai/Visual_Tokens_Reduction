# Visual Token Reduction

This folder stages a ProtoReduce experiment for comparing vanilla LLaVA 13B
against prototype-reduced visual tokens on POPE adversarial.

## Files

- `proto_reduce.py`: standalone PyTorch implementation of ProtoReduce.
- `run_llava13b_pope.py`: LLaVA 13B POPE runner with `vanilla` and
  `proto_reduce` methods.
- `ALGORITHM.md`: detailed explanation of the current ProtoReduce algorithm
  and how it is inserted into LLaVA 13B.
- `requirements.txt`: Python dependencies for running the experiment.

## Quick Run

```bash
python Visual_Token_Reduction/run_llava13b_pope.py \
  --model-id llava-hf/llava-1.5-13b-hf \
  --methods vanilla,proto_reduce \
  --prototype-tokens 128 \
  --cluster-iters 5 \
  --temperature 0.07 \
  --limit 100 \
  --output-dir results/llava13b_proto_reduce_100
```

For a local checkpoint, replace `--model-id` with the checkpoint path.

If the 13B model does not fit, use quantization:

```bash
python Visual_Token_Reduction/run_llava13b_pope.py \
  --model-id llava-hf/llava-1.5-13b-hf \
  --load-in-4bit \
  --methods vanilla,proto_reduce \
  --prototype-tokens 128 \
  --limit 100
```

## What Is Compared

The script writes one JSONL per method plus `summary.json`. When both methods
are enabled, the summary includes `delta_proto_minus_vanilla` for accuracy,
precision, recall, F1, and yes ratio.

ProtoReduce is applied after LLaVA's vision tower and multimodal projector
produce image features. The script also adjusts the number of image token
placeholders in the text input so the reduced feature count matches the model's
expected image-token count.
