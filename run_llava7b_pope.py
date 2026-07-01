from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from foresight import ForesightConfig, foresight_reduce


@dataclass(frozen=True)
class POPEExample:
    image_id: str
    question_id: str
    question: str
    label: str
    image: Any
    category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run FORESIGHT on LLaVA-1.5-7B and POPE.")
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--dataset-name", default="lmms-lab/POPE")
    parser.add_argument("--split", default="test")
    parser.add_argument("--category", default="all")
    parser.add_argument("--output-dir", default="results/llava7b_pope_foresight")
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--sampling", choices=["first", "shuffle"], default="first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-min", type=int, default=32)
    parser.add_argument("--k-max", type=int, default=128)
    parser.add_argument("--k-text", type=int, default=64)
    parser.add_argument("--rho", type=float, default=0.90)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[POPE] FORESIGHT enabled: model={args.model_id}, dataset={args.dataset_name}, "
        f"split={args.split}, category={args.category}, K_min={args.k_min}, "
        f"K_max={args.k_max}, K_text={args.k_text}, rho={args.rho}",
        flush=True,
    )

    examples = load_pope_examples(
        dataset_name=args.dataset_name,
        split=args.split,
        category=args.category,
        limit_questions=args.limit_questions,
        sampling=args.sampling,
        seed=args.seed,
    )
    print(f"[POPE] Loaded {len(examples)} questions.", flush=True)

    runner = LlavaPOPERunner(
        model_id=args.model_id,
        device_map=args.device_map,
        dtype=args.dtype,
        load_in_4bit=bool(args.load_in_4bit),
        k_config=ForesightConfig(
            k_min=int(args.k_min),
            k_max=int(args.k_max),
            k_text=int(args.k_text),
            rho=float(args.rho),
            eps=float(args.eps),
        ),
        max_new_tokens=int(args.max_new_tokens),
    )
    print("[POPE] Model ready.", flush=True)

    jsonl_path = output_dir / "foresight.jsonl"
    csv_path = output_dir / "per_sample_efficiency.csv"
    summary_path = output_dir / "summary.json"
    csv_fields = [
        "index",
        "image_id",
        "question_id",
        "category",
        "label",
        "prediction",
        "correct",
        "total_sec",
        "questions_per_sec",
        "generation_sec",
        "generated_tokens_per_sec",
        "original_image_tokens",
        "reduced_image_tokens",
        "visual_token_keep_ratio",
        "visual_token_reduction_ratio",
        "k_eff",
        "k_rho",
        "k_budget",
        "iterations_used",
        "estimated_total_flops",
        "estimated_llm_prefill_flops",
        "estimated_llm_decode_flops",
        "estimated_reduction_flops",
        "gpu_peak_mem_mb",
    ]

    all_rows: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file, csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

        for index, example in enumerate(examples, start=1):
            result = runner.ask_yes_no(example.image, example.question)
            row = {
                "image_id": example.image_id,
                "question_id": example.question_id,
                "category": example.category,
                "question": example.question,
                "label": example.label,
                "method": "foresight",
                "prediction": result["prediction"],
                "raw_text": result["raw_text"],
                "confidence": result["confidence"],
                "correct": result["prediction"] == example.label,
                "elapsed_sec": result["meta"]["total_sec"],
                "meta": result["meta"],
            }
            all_rows.append(row)
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl_file.flush()

            csv_row = {field: row["meta"].get(field, row.get(field)) for field in csv_fields}
            csv_row["index"] = index
            csv_row["image_id"] = example.image_id
            csv_row["question_id"] = example.question_id
            csv_row["category"] = example.category
            csv_row["label"] = example.label
            csv_row["prediction"] = result["prediction"]
            csv_row["correct"] = row["correct"]
            writer.writerow(csv_row)
            csv_file.flush()

            meta = result["meta"]
            print(
                f"[{index}/{len(examples)}] image={example.image_id} category={example.category} "
                f"label={example.label} pred={result['prediction']} correct={row['correct']} "
                f"time={meta['total_sec']:.2f}s vis={meta['original_image_tokens']}->"
                f"{meta['reduced_image_tokens']}({meta['visual_token_keep_ratio']:.2f}) "
                f"k_eff={meta['k_eff']} qps={meta['questions_per_sec']:.3f} "
                f"flops={format_flops(meta['estimated_total_flops'])} "
                f"mem={meta['gpu_peak_mem_mb']:.0f}MB",
                flush=True,
            )

    summary = summarize_results(all_rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(summary)
    print(f"[POPE] Wrote results to: {output_dir}", flush=True)


class LlavaPOPERunner:
    def __init__(
        self,
        model_id: str,
        device_map: str,
        dtype: str,
        load_in_4bit: bool,
        k_config: ForesightConfig,
        max_new_tokens: int,
    ) -> None:
        from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_id)
        quantization_config = None
        torch_dtype = resolve_dtype(dtype)
        load_kwargs: dict[str, Any] = {"device_map": device_map}
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["torch_dtype"] = torch_dtype

        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        self.k_config = k_config
        self.max_new_tokens = int(max_new_tokens)
        self.image_token_id = self._image_token_id()
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        self.yes_token_ids = self._candidate_first_token_ids(["yes", "Yes", " yes", " Yes"])
        self.no_token_ids = self._candidate_first_token_ids(["no", "No", " no", " No"])
        self.text_layers, self.text_hidden, self.text_intermediate = self._text_model_shape()
        self._current_question = ""
        self._last_reduction_meta: dict[str, Any] = {}

    def ask_yes_no(self, image: Any, question: str) -> dict[str, Any]:
        total_start = perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self._current_question = question
        prompt = build_prompt(self.processor, question)
        preprocess_start = perf_counter()
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = move_inputs_to_model(inputs, self.model)
        preprocess_sec = perf_counter() - preprocess_start

        reduction_start = perf_counter()
        reduced_features = self._precompute_reduced_image_features(inputs)
        feature_and_reduction_sec = perf_counter() - reduction_start
        dynamic_tokens = image_feature_count(reduced_features)
        inputs = force_image_placeholder_count(
            inputs,
            image_token_id=self.image_token_id,
            target_count=dynamic_tokens,
            pad_token_id=int(self.pad_token_id),
        )

        with self._precomputed_image_features_context(reduced_features):
            confidence_start = perf_counter()
            confidence = self._yes_no_confidence(inputs)
            confidence_sec = perf_counter() - confidence_start

            generation_start = perf_counter()
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_sec = perf_counter() - generation_start

        prompt_len = int(inputs["input_ids"].shape[-1])
        generated_tokens = max(0, int(generated.shape[-1]) - prompt_len)
        raw_text = self.processor.decode(
            generated[0][prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        prediction = normalize_yes_no(raw_text)
        if prediction == "unknown":
            prediction = "yes" if confidence["yes"] >= confidence["no"] else "no"

        total_sec = perf_counter() - total_start
        total_input_tokens = int(inputs["input_ids"].shape[-1])
        visual_tokens = count_image_placeholders(inputs, self.image_token_id)
        llm_prefill_flops = estimate_llm_prefill_flops(
            seq_len=total_input_tokens,
            num_layers=self.text_layers,
            hidden_size=self.text_hidden,
            intermediate_size=self.text_intermediate,
        )
        llm_decode_flops = estimate_llm_decode_flops(
            prompt_len=total_input_tokens,
            generated_tokens=generated_tokens,
            num_layers=self.text_layers,
            hidden_size=self.text_hidden,
            intermediate_size=self.text_intermediate,
        )
        reduction_flops = float(self._last_reduction_meta.get("reduction_flops", 0.0))
        peak_mem = (
            float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            if torch.cuda.is_available()
            else 0.0
        )
        meta = {
            "algorithm": "foresight",
            "algorithm_name": "FORESIGHT",
            "yes_prob": confidence["yes"],
            "no_prob": confidence["no"],
            "preprocess_sec": preprocess_sec,
            "feature_and_reduction_sec": feature_and_reduction_sec,
            "confidence_sec": confidence_sec,
            "generation_sec": generation_sec,
            "total_sec": total_sec,
            "questions_per_sec": 1.0 / total_sec if total_sec > 0 else 0.0,
            "generated_tokens": generated_tokens,
            "generated_tokens_per_sec": generated_tokens / generation_sec if generation_sec > 0 else 0.0,
            "total_input_tokens": total_input_tokens,
            "visual_input_tokens": visual_tokens,
            "text_input_tokens_estimate": total_input_tokens - visual_tokens,
            "estimated_llm_prefill_flops": llm_prefill_flops,
            "estimated_llm_decode_flops": llm_decode_flops,
            "estimated_reduction_flops": reduction_flops,
            "estimated_total_flops": llm_prefill_flops + llm_decode_flops + reduction_flops,
            "gpu_peak_mem_mb": peak_mem,
            **self._last_reduction_meta,
            "dynamic_image_hypotheses": True,
            "dynamic_text_hypotheses": True,
            "fixed_semantic_bank": False,
            "reduction_stage": "after_vision_tower_and_projector",
            "k_min": self.k_config.k_min,
            "k_max": self.k_config.k_max,
            "k_text": self.k_config.k_text,
            "rho": self.k_config.rho,
        }
        if "original_image_tokens" in meta and total_sec > 0:
            meta["original_visual_tokens_per_sec"] = float(meta["original_image_tokens"]) / total_sec
            meta["reduced_visual_tokens_per_sec"] = float(meta["reduced_image_tokens"]) / total_sec
        return {
            "prediction": prediction,
            "raw_text": raw_text,
            "confidence": max(confidence["yes"], confidence["no"]),
            "meta": meta,
        }

    def _precompute_reduced_image_features(self, inputs: dict[str, torch.Tensor]) -> Any:
        features = self._call_get_image_features(inputs)
        return self._reduce_image_features(features)

    def _call_get_image_features(self, inputs: dict[str, torch.Tensor]) -> Any:
        modules = self._image_feature_modules()
        if not modules:
            raise ValueError("The model does not expose get_image_features")
        method = modules[-1].get_image_features
        kwargs: dict[str, Any] = {}
        import inspect

        signature = inspect.signature(method)
        for name in signature.parameters:
            if name == "pixel_values" and "pixel_values" in inputs:
                kwargs[name] = inputs["pixel_values"]
            elif name == "vision_feature_layer":
                kwargs[name] = getattr(self.model.config, "vision_feature_layer", -2)
            elif name == "vision_feature_select_strategy":
                kwargs[name] = getattr(self.model.config, "vision_feature_select_strategy", "default")
            elif name == "image_sizes" and "image_sizes" in inputs:
                kwargs[name] = inputs["image_sizes"]
        try:
            with torch.inference_mode():
                return method(**kwargs)
        except TypeError:
            with torch.inference_mode():
                return method(inputs["pixel_values"])

    @contextmanager
    def _precomputed_image_features_context(self, precomputed_features: Any):
        modules = self._image_feature_modules()
        patched: list[tuple[Any, Any]] = []

        def patched_get_image_features(*args: Any, **kwargs: Any) -> Any:
            return precomputed_features

        for module in modules:
            patched.append((module, module.get_image_features))
            module.get_image_features = patched_get_image_features
        try:
            yield
        finally:
            for module, original in patched:
                module.get_image_features = original

    def _image_feature_modules(self) -> list[Any]:
        modules = []
        for candidate in (self.model, getattr(self.model, "model", None)):
            if candidate is not None and hasattr(candidate, "get_image_features"):
                modules.append(candidate)
        return modules

    def _reduce_image_features(self, features: Any) -> Any:
        if torch.is_tensor(features):
            if features.ndim in {2, 3}:
                return self._reduce_tensor_features(features)
            return features
        if hasattr(features, "pooler_output"):
            features.pooler_output = self._reduce_image_features(features.pooler_output)
            return features
        if isinstance(features, list):
            return [self._reduce_image_features(item) for item in features]
        if isinstance(features, tuple):
            return tuple(self._reduce_image_features(item) for item in features)
        return features

    def _reduce_tensor_features(self, features: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if features.ndim == 3:
            if int(features.shape[0]) != 1:
                raise ValueError("This runner expects batch size 1 for variable visual tokens")
            tokens = features[0]
            squeeze_batch = True
        else:
            tokens = features

        original_tokens = int(tokens.shape[0])
        hidden_dim = int(tokens.shape[-1])
        text_hypotheses = self._text_hypotheses_for_question(
            question=self._current_question,
            hidden_dim=hidden_dim,
            device=tokens.device,
        )
        result = foresight_reduce(tokens, text_hypotheses=text_hypotheses, config=self.k_config)
        reduced = result.patch_tokens
        text_count = 0 if text_hypotheses is None else int(text_hypotheses.shape[0])
        reduction_flops = estimate_foresight_flops(
            original_tokens=original_tokens,
            hidden_dim=hidden_dim,
            candidate_pool_size=original_tokens + text_count,
            k_budget=int(result.k_budget),
            k_eff=int(result.k_eff),
            iterations=int(result.iterations_used),
        )
        self._last_reduction_meta = {
            "original_image_tokens": original_tokens,
            "reduced_image_tokens": int(reduced.shape[-2]),
            "visual_token_keep_ratio": float(reduced.shape[-2]) / max(1, original_tokens),
            "visual_token_reduction_ratio": float(original_tokens) / max(1, int(reduced.shape[-2])),
            "reduction_flops": reduction_flops,
            "reduction_strategy": "foresight_dynamic_hypothesis_reduction",
            "hypothesis_generation": "per_image_token_residual_plus_text_embeddings",
            "candidate_selection": "omega_weighted_diversity_selection",
            "evidence_rule": "multiplicative_image_confidence_and_token_support",
            "image_hypothesis_count": int(result.image_hypothesis_count),
            "text_hypothesis_count": int(result.text_hypothesis_count),
            "candidate_hypothesis_count": int(result.candidate_indices.numel()),
            "active_hypothesis_count": int(result.hypothesis_indices.numel()),
            "k_min_effective": int(result.k_min),
            "k_max_requested": int(result.k_max),
            "k_budget": int(result.k_budget),
            "k_rho": int(result.k_rho),
            "k_eff": int(result.k_eff),
            "iterations_used": int(result.iterations_used),
            "final_assignment_change_rate": result.final_assignment_change_rate,
            "mean_region_mass": float(result.masses.float().mean().item()),
            "max_region_mass": int(result.masses.max().item()),
            "min_region_mass": int(result.masses.min().item()),
        }
        if squeeze_batch:
            return reduced.unsqueeze(0)
        return reduced

    def _text_hypotheses_for_question(self, question: str, hidden_dim: int, device: torch.device) -> torch.Tensor | None:
        if int(self.k_config.k_text) <= 0:
            return None
        token_ids = self.tokenizer.encode(question, add_special_tokens=False)
        special_ids = {
            value
            for value in [
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.unk_token_id,
                self.image_token_id,
            ]
            if value is not None
        }
        token_ids = [int(token_id) for token_id in token_ids if int(token_id) not in special_ids]
        if not token_ids:
            return None
        embeddings = self.model.get_input_embeddings()
        if embeddings is None or not hasattr(embeddings, "weight"):
            return None
        weight = embeddings.weight.detach()
        if int(weight.shape[-1]) != int(hidden_dim):
            return None
        ids = torch.tensor(token_ids[: int(self.k_config.k_text)], dtype=torch.long, device=weight.device)
        return weight.index_select(0, ids).to(device=device, dtype=torch.float32)

    def _yes_no_confidence(self, inputs: dict[str, torch.Tensor]) -> dict[str, float]:
        with torch.inference_mode():
            outputs = self.model(**inputs)
        logits = outputs.logits[0, -1].float()
        yes = torch.logsumexp(logits[self.yes_token_ids], dim=0)
        no = torch.logsumexp(logits[self.no_token_ids], dim=0)
        probs = torch.softmax(torch.stack([yes, no]), dim=0)
        return {"yes": float(probs[0].item()), "no": float(probs[1].item())}

    def _candidate_first_token_ids(self, candidates: list[str]) -> torch.Tensor:
        ids = []
        for candidate in candidates:
            encoded = self.tokenizer.encode(candidate, add_special_tokens=False)
            if encoded:
                ids.append(int(encoded[0]))
        if not ids:
            raise ValueError("Could not build yes/no candidate token ids")
        unique = sorted(set(ids))
        return torch.tensor(unique, dtype=torch.long, device=model_device(self.model))

    def _image_token_id(self) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        if token_id is None or int(token_id) < 0:
            token_id = getattr(self.model.config, "image_token_index", None)
        if token_id is None:
            raise ValueError("Could not determine LLaVA image token id")
        return int(token_id)

    def _text_model_shape(self) -> tuple[int, int, int]:
        config = getattr(self.model.config, "text_config", self.model.config)
        layers = int(getattr(config, "num_hidden_layers", 0) or getattr(config, "num_layers", 0) or 0)
        hidden = int(getattr(config, "hidden_size", 0) or getattr(config, "d_model", 0) or 0)
        intermediate = int(getattr(config, "intermediate_size", 0) or (4 * hidden if hidden else 0))
        return layers, hidden, intermediate


def load_pope_examples(
    dataset_name: str,
    split: str,
    category: str,
    limit_questions: int | None,
    sampling: str,
    seed: int,
) -> list[POPEExample]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    examples: list[POPEExample] = []
    wanted = category.lower()
    for index, row in enumerate(dataset):
        row_category = str(row.get("category") or row.get("type") or row.get("subset") or "unknown").lower()
        if wanted != "all" and row_category != wanted:
            continue
        question = str(row.get("question") or row.get("query") or row.get("text") or "")
        label = normalize_yes_no(str(row.get("label") or row.get("answer") or row.get("gt_answer") or ""))
        if label == "unknown":
            continue
        image = row.get("image")
        if image is None:
            continue
        image_id = str(row.get("image_id") or row.get("image") or index)
        question_id = str(row.get("question_id") or row.get("id") or index)
        examples.append(
            POPEExample(
                image_id=image_id,
                question_id=question_id,
                question=question,
                label=label,
                image=image,
                category=row_category,
            )
        )
    if sampling == "shuffle":
        rng = random.Random(int(seed))
        rng.shuffle(examples)
    if limit_questions is not None:
        examples = examples[: int(limit_questions)]
    return examples


def build_prompt(processor: Any, question: str) -> str:
    if hasattr(processor, "apply_chat_template"):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        return processor.apply_chat_template(conversation, add_generation_prompt=True)
    return f"USER: <image>\n{question}\nASSISTANT:"


def move_inputs_to_model(inputs: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, torch.Tensor]:
    device = model_device(model)
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def model_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(getattr(model, "device"))
    return next(model.parameters()).device


def count_image_placeholders(inputs: dict[str, torch.Tensor], image_token_id: int) -> int:
    return int((inputs["input_ids"] == int(image_token_id)).sum().item())


def force_image_placeholder_count(
    inputs: dict[str, torch.Tensor],
    image_token_id: int,
    target_count: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    current = count_image_placeholders(inputs, image_token_id)
    if current == int(target_count):
        return inputs
    if int(inputs["input_ids"].shape[0]) != 1:
        raise ValueError("This runner expects batch size 1")
    ids = inputs["input_ids"][0]
    positions = torch.nonzero(ids == int(image_token_id), as_tuple=False).flatten()
    if int(positions.numel()) == 0:
        raise ValueError("No image placeholders found in input_ids")
    first = int(positions[0].item())
    last = int(positions[-1].item())
    replacement = torch.full((int(target_count),), int(image_token_id), dtype=ids.dtype, device=ids.device)
    new_ids = torch.cat([ids[:first], replacement, ids[last + 1 :]], dim=0).unsqueeze(0)
    new_attention = torch.ones_like(new_ids)
    updated = dict(inputs)
    updated["input_ids"] = new_ids
    updated["attention_mask"] = new_attention
    if "position_ids" in updated:
        del updated["position_ids"]
    if "token_type_ids" in updated:
        updated["token_type_ids"] = torch.zeros_like(new_ids)
    _ = pad_token_id
    return updated


def image_feature_count(features: Any) -> int:
    if torch.is_tensor(features):
        if features.ndim == 3:
            return int(features.shape[-2])
        if features.ndim == 2:
            return int(features.shape[0])
    if hasattr(features, "pooler_output"):
        return image_feature_count(features.pooler_output)
    if isinstance(features, (list, tuple)) and features:
        return image_feature_count(features[0])
    raise ValueError("Could not infer image feature count")


def normalize_yes_no(text: str) -> str:
    value = text.strip().lower()
    if value.startswith("yes"):
        return "yes"
    if value.startswith("no"):
        return "no"
    if value in {"true", "1"}:
        return "yes"
    if value in {"false", "0"}:
        return "no"
    return "unknown"


def resolve_dtype(dtype: str) -> str | torch.dtype:
    if dtype == "auto":
        return "auto"
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def estimate_llm_prefill_flops(seq_len: int, num_layers: int, hidden_size: int, intermediate_size: int) -> float:
    if seq_len <= 0 or num_layers <= 0 or hidden_size <= 0:
        return 0.0
    attention_proj = 8.0 * seq_len * hidden_size * hidden_size
    attention_scores = 4.0 * seq_len * seq_len * hidden_size
    mlp = 6.0 * seq_len * hidden_size * max(1, intermediate_size)
    return float(num_layers) * (attention_proj + attention_scores + mlp)


def estimate_llm_decode_flops(
    prompt_len: int,
    generated_tokens: int,
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
) -> float:
    total = 0.0
    for step in range(max(0, int(generated_tokens))):
        total += estimate_llm_prefill_flops(1, num_layers, hidden_size, intermediate_size)
        total += float(num_layers) * 4.0 * float(prompt_len + step) * float(hidden_size)
    return total


def estimate_foresight_flops(
    original_tokens: int,
    hidden_dim: int,
    candidate_pool_size: int,
    k_budget: int,
    k_eff: int,
    iterations: int,
) -> float:
    n = float(original_tokens)
    d = float(hidden_dim)
    m = float(candidate_pool_size)
    kb = float(k_budget)
    ke = float(k_eff)
    iters = float(iterations)
    normalize_and_residual = 8.0 * n * d
    support = 2.0 * m * d + 2.0 * n * m * d
    candidate_response = 2.0 * n * kb * d
    active_response = 2.0 * n * ke * d
    clustering = iters * (2.0 * n * ke * ke + 4.0 * n * ke)
    aggregation = 2.0 * n * d
    return float(normalize_and_residual + support + candidate_response + active_response + clustering + aggregation)


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    summary = {
        "overall": compute_metrics(rows),
        "by_category": {category: compute_metrics(items) for category, items in sorted(by_category.items())},
        "num_questions": len(rows),
    }
    return summary


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    tp = sum(1 for r in rows if r["label"] == "yes" and r["prediction"] == "yes")
    fp = sum(1 for r in rows if r["label"] == "no" and r["prediction"] == "yes")
    tn = sum(1 for r in rows if r["label"] == "no" and r["prediction"] == "no")
    fn = sum(1 for r in rows if r["label"] == "yes" and r["prediction"] == "no")
    total = len(rows)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    metas = [r["meta"] for r in rows]
    return {
        "questions": total,
        "accuracy": (tp + tn) / max(1, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_rate": (tp + fp) / max(1, total),
        "avg_sec": mean([m["total_sec"] for m in metas]),
        "questions_per_sec": 1.0 / mean([m["total_sec"] for m in metas]),
        "avg_original_image_tokens": mean([m["original_image_tokens"] for m in metas]),
        "avg_reduced_image_tokens": mean([m["reduced_image_tokens"] for m in metas]),
        "avg_keep_ratio": mean([m["visual_token_keep_ratio"] for m in metas]),
        "avg_total_flops": mean([m["estimated_total_flops"] for m in metas]),
        "avg_gpu_peak_mem_mb": mean([m["gpu_peak_mem_mb"] for m in metas]),
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def print_summary(summary: dict[str, Any]) -> None:
    overall = summary.get("overall", {})
    print("[POPE] Overall", flush=True)
    print(
        "acc={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} "
        "f1={f1:.4f} yes_rate={yes_rate:.4f} vis_avg={avg_reduced_image_tokens:.1f} "
        "qps={questions_per_sec:.3f}".format(**overall),
        flush=True,
    )


def format_flops(value: float) -> str:
    if value >= 1e12:
        return f"{value / 1e12:.2f}T"
    if value >= 1e9:
        return f"{value / 1e9:.2f}G"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    return f"{value:.0f}"


if __name__ == "__main__":
    main()
