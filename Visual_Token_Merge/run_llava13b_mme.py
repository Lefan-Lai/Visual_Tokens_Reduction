from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F

from early_semreduce import (
    AnswerAdaptiveKSemReduceConfig,
    KSemReduceConfig,
    answer_adaptive_k_semreduce,
    k_semreduce,
)


PERCEPTION_CATEGORIES = {
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
}
COGNITION_CATEGORIES = {
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "please",
    "shown",
    "the",
    "there",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "will",
    "with",
    "yes",
    "no",
    "answer",
    "picture",
    "image",
}

RELATION_PHRASES = [
    "above",
    "behind",
    "below",
    "beside",
    "between",
    "holding",
    "inside",
    "near",
    "next to",
    "on top of",
    "under",
    "wearing",
]

COLORS = [
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
]

KNOWN_VISUAL_PHRASES = [
    "baseball bat",
    "baseball glove",
    "cell phone",
    "dining table",
    "fire hydrant",
    "hot dog",
    "parking meter",
    "sports ball",
    "stop sign",
    "tennis racket",
    "traffic light",
    "wine glass",
]

SEMANTIC_VOCABULARY = [
    "person",
    "people",
    "man",
    "woman",
    "child",
    "face",
    "head",
    "hand",
    "body",
    "animal",
    "dog",
    "cat",
    "bird",
    "horse",
    "sheep",
    "cow",
    "car",
    "bus",
    "truck",
    "train",
    "bicycle",
    "motorcycle",
    "airplane",
    "boat",
    "table",
    "chair",
    "sofa",
    "bed",
    "screen",
    "computer",
    "phone",
    "book",
    "cup",
    "bottle",
    "bowl",
    "plate",
    "food",
    "fruit",
    "apple",
    "banana",
    "umbrella",
    "bag",
    "clock",
    "text",
    "word",
    "letter",
    "number",
    "sign",
    "poster",
    "code",
    "python",
    "program",
    "output",
    "painting",
    "artwork",
    "landmark",
    "building",
    "street",
    "room",
    "scene",
    "color",
    "position",
    "count",
    "near relation",
    "holding relation",
    "wearing relation",
    "inside relation",
]

CATEGORY_HINTS = {
    "OCR": ["text", "word", "letter", "number", "sign"],
    "artwork": ["artwork", "painting", "picture", "style"],
    "celebrity": ["person", "face", "celebrity"],
    "code_reasoning": ["code", "python", "program", "output", "number"],
    "color": ["color"],
    "commonsense_reasoning": ["object", "scene", "relation"],
    "count": ["count", "number", "object"],
    "existence": ["object", "presence"],
    "landmark": ["landmark", "building", "place"],
    "numerical_calculation": ["number", "calculation", "text"],
    "position": ["position", "relation", "location"],
    "posters": ["poster", "text", "image"],
    "scene": ["scene", "place", "background"],
    "text_translation": ["text", "word", "translation", "language"],
}


@dataclass(frozen=True)
class MMEExample:
    image_id: str
    question_id: str
    question: str
    label: str
    image: Any
    category: str
    dimension: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate vanilla LLaVA 13B and Answer-Adaptive K-SemReduce on MME."
    )
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-13b-hf")
    parser.add_argument("--dataset-name", default="lmms-lab/MME")
    parser.add_argument("--split", default="test")
    parser.add_argument("--methods", default="vanilla,answer_adaptive_k_semreduce")
    parser.add_argument("--limit-images", type=int, default=300)
    parser.add_argument("--category", default="")
    parser.add_argument("--sampling", choices=["stratified", "shuffle", "first"], default="stratified")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/llava13b_answer_adaptive_k_semreduce_mme300")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--candidate-classes", type=int, default=64)
    parser.add_argument("--k-cluster-iters", type=int, default=3)
    parser.add_argument("--k-temperature", type=float, default=0.1)
    parser.add_argument("--k-lambda-importance", type=float, default=0.25)
    parser.add_argument("--k-gamma", type=float, default=1.0)
    parser.add_argument("--adaptive-cluster-iters", type=int, default=3)
    parser.add_argument("--adaptive-temperature", type=float, default=0.1)
    parser.add_argument("--adaptive-lambda-importance", type=float, default=0.25)
    parser.add_argument("--adaptive-gamma", type=float, default=1.0)
    parser.add_argument("--adaptive-hypothesis-multiplier", type=int, default=1)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    known_methods = {"vanilla", "k_semreduce", "answer_adaptive_k_semreduce"}
    unknown = [method for method in methods if method not in known_methods]
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(unknown)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MME] Loading dataset {args.dataset_name}/{args.split}...")
    examples = load_mme_examples(
        dataset_name=args.dataset_name,
        split=args.split,
        limit_images=args.limit_images,
        seed=args.seed,
        sampling=args.sampling,
        category=args.category,
    )
    image_count = len({example.image_id for example in examples})
    print(f"[MME] Loaded {len(examples)} questions from {image_count} images.")

    print(f"[MME] Loading model: {args.model_id}")
    runner = LlavaMMERunner(
        model_id=args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        load_in_4bit=args.load_in_4bit,
        candidate_classes=args.candidate_classes,
        k_cluster_iters=args.k_cluster_iters,
        k_temperature=args.k_temperature,
        k_lambda_importance=args.k_lambda_importance,
        k_gamma=args.k_gamma,
        adaptive_cluster_iters=args.adaptive_cluster_iters,
        adaptive_temperature=args.adaptive_temperature,
        adaptive_lambda_importance=args.adaptive_lambda_importance,
        adaptive_gamma=args.adaptive_gamma,
        adaptive_hypothesis_multiplier=args.adaptive_hypothesis_multiplier,
    )
    print("[MME] Model ready. Running methods:", ", ".join(methods))

    records_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    writers = {
        method: (output_dir / f"{method}.jsonl").open("w", encoding="utf-8")
        for method in methods
    }

    start = perf_counter()
    try:
        for index, example in enumerate(examples, start=1):
            print(
                f"\n[{index}/{len(examples)}] image={example.image_id} "
                f"category={example.category} label={example.label}"
            )
            print(f"  q={example.question}")
            for method in methods:
                method_start = perf_counter()
                answer = runner.ask_yes_no(
                    image=example.image,
                    question=example.question,
                    method=method,
                    category=example.category,
                )
                elapsed = perf_counter() - method_start
                record = {
                    "image_id": example.image_id,
                    "question_id": example.question_id,
                    "category": example.category,
                    "dimension": example.dimension,
                    "question": example.question,
                    "label": example.label,
                    "method": method,
                    "prediction": answer["prediction"],
                    "raw_text": answer["raw_text"],
                    "confidence": answer["confidence"],
                    "correct": answer["prediction"] == example.label,
                    "elapsed_sec": elapsed,
                    "meta": answer["meta"],
                }
                records_by_method[method].append(record)
                writers[method].write(json.dumps(record, ensure_ascii=False) + "\n")
                writers[method].flush()
                print(
                    f"  - {method:<30} pred={record['prediction']:<3} "
                    f"conf={record['confidence']:.3f} correct={record['correct']} "
                    f"time={elapsed:.1f}s"
                )
    finally:
        for writer in writers.values():
            writer.close()

    summary = {
        method: summarize_mme(records)
        for method, records in records_by_method.items()
    }
    if "vanilla" in summary:
        for method in methods:
            if method == "vanilla":
                continue
            summary[f"delta_{method}_minus_vanilla"] = delta_overall(
                summary[method]["overall"],
                summary["vanilla"]["overall"],
            )
    summary["run"] = {
        "model_id": args.model_id,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "limit_images": args.limit_images,
        "actual_images": image_count,
        "actual_questions": len(examples),
        "category": args.category,
        "sampling": args.sampling,
        "seed": args.seed,
        "methods": methods,
        "candidate_classes": args.candidate_classes,
        "k_cluster_iters": args.k_cluster_iters,
        "k_temperature": args.k_temperature,
        "k_lambda_importance": args.k_lambda_importance,
        "k_gamma": args.k_gamma,
        "adaptive_cluster_iters": args.adaptive_cluster_iters,
        "adaptive_temperature": args.adaptive_temperature,
        "adaptive_lambda_importance": args.adaptive_lambda_importance,
        "adaptive_gamma": args.adaptive_gamma,
        "adaptive_hypothesis_multiplier": args.adaptive_hypothesis_multiplier,
        "load_in_4bit": args.load_in_4bit,
        "elapsed_sec": perf_counter() - start,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_metric_csvs(output_dir, summary, methods)

    print("\n[MME] Overall")
    print(format_overall_table(summary, methods))
    print("\n[MME] Dimensions")
    print(format_dimension_table(summary, methods))
    print(f"\n[MME] Wrote results to: {output_dir.resolve()}")


class LlavaMMERunner:
    def __init__(
        self,
        model_id: str,
        dtype: str,
        device_map: str,
        max_new_tokens: int,
        load_in_4bit: bool,
        candidate_classes: int,
        k_cluster_iters: int,
        k_temperature: float,
        k_lambda_importance: float,
        k_gamma: float,
        adaptive_cluster_iters: int,
        adaptive_temperature: float,
        adaptive_lambda_importance: float,
        adaptive_gamma: float,
        adaptive_hypothesis_multiplier: int,
    ) -> None:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            model_kwargs["torch_dtype"] = resolve_dtype(dtype)

        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        self.max_new_tokens = max_new_tokens
        self.k_config = KSemReduceConfig(
            num_semantic_classes=candidate_classes,
            iterations=k_cluster_iters,
            temperature=k_temperature,
            lambda_importance=k_lambda_importance,
            gamma=k_gamma,
        )
        self.adaptive_config = AnswerAdaptiveKSemReduceConfig(
            iterations=adaptive_cluster_iters,
            temperature=adaptive_temperature,
            lambda_importance=adaptive_lambda_importance,
            gamma=adaptive_gamma,
        )
        self.adaptive_hypothesis_multiplier = max(1, int(adaptive_hypothesis_multiplier))
        self.image_token_id = self._image_token_id()
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        self.yes_token_ids = self._candidate_first_token_ids(["yes", "Yes", " yes", " Yes"])
        self.no_token_ids = self._candidate_first_token_ids(["no", "No", " no", " No"])
        self._classifier_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._embedding_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._semantic_vocab_cache: dict[tuple[str, int], tuple[list[str], torch.Tensor]] = {}
        self._last_reduction_meta: dict[str, Any] = {}

    def ask_yes_no(self, image: Any, question: str, method: str, category: str) -> dict[str, Any]:
        prompt = build_prompt(self.processor, question)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = move_inputs_to_model(inputs, self.model)
        self._last_reduction_meta = {}

        if method in {"k_semreduce", "answer_adaptive_k_semreduce"}:
            reduced_features = self._precompute_dynamic_image_features(inputs, method, question, category)
            dynamic_tokens = image_feature_count(reduced_features)
            inputs = force_image_placeholder_count(
                inputs,
                image_token_id=self.image_token_id,
                target_count=dynamic_tokens,
                pad_token_id=int(self.pad_token_id),
            )
            context = self._precomputed_image_features_context(reduced_features)
        else:
            context = nullcontext()

        with context:
            confidence = self._yes_no_confidence(inputs)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

        prompt_len = int(inputs["input_ids"].shape[-1])
        text = self.processor.decode(
            generated[0][prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        prediction = normalize_yes_no(text)
        meta: dict[str, Any] = {
            "algorithm": method,
            "yes_prob": confidence["yes"],
            "no_prob": confidence["no"],
        }
        if method == "k_semreduce":
            meta.update(
                {
                    "K": self.k_config.num_semantic_classes,
                    "cluster_iters": self.k_config.iterations,
                    "temperature": self.k_config.temperature,
                    "lambda_importance": self.k_config.lambda_importance,
                    "gamma": self.k_config.gamma,
                    **self._last_reduction_meta,
                }
            )
        elif method == "answer_adaptive_k_semreduce":
            meta.update(
                {
                    "adaptive_hypothesis_multiplier": self.adaptive_hypothesis_multiplier,
                    "cluster_iters": self.adaptive_config.iterations,
                    "temperature": self.adaptive_config.temperature,
                    "lambda_importance": self.adaptive_config.lambda_importance,
                    "gamma": self.adaptive_config.gamma,
                    **self._last_reduction_meta,
                }
            )
        return {
            "prediction": prediction,
            "raw_text": text,
            "confidence": confidence[prediction],
            "meta": meta,
        }

    def _yes_no_confidence(self, inputs: dict[str, torch.Tensor]) -> dict[str, float]:
        with torch.inference_mode():
            logits = self.model(**inputs).logits[:, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        yes = probs[:, self.yes_token_ids].sum(dim=-1) if self.yes_token_ids else torch.tensor([0.0])
        no = probs[:, self.no_token_ids].sum(dim=-1) if self.no_token_ids else torch.tensor([0.0])
        total = yes + no
        if float(total.item()) <= 0:
            return {"yes": 0.5, "no": 0.5}
        return {
            "yes": float((yes / total).item()),
            "no": float((no / total).item()),
        }

    def _candidate_first_token_ids(self, texts: list[str]) -> list[int]:
        token_ids: list[int] = []
        for text in texts:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                token_ids.append(int(encoded[0]))
        return sorted(set(token_ids))

    def _image_token_id(self) -> int:
        if hasattr(self.model.config, "image_token_index"):
            return int(self.model.config.image_token_index)
        token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise ValueError("Could not resolve the LLaVA image token id")
        return int(token_id)

    def _semantic_classifier_for(self, device: torch.device, hidden_dim: int) -> torch.Tensor:
        cache_key = (str(device), int(hidden_dim))
        if cache_key in self._classifier_cache:
            return self._classifier_cache[cache_key]

        candidates = []
        output_embeddings = self.model.get_output_embeddings()
        input_embeddings = self.model.get_input_embeddings()
        if output_embeddings is not None and hasattr(output_embeddings, "weight"):
            candidates.append(output_embeddings.weight)
        if input_embeddings is not None and hasattr(input_embeddings, "weight"):
            candidates.append(input_embeddings.weight)

        for weight in candidates:
            if int(weight.shape[-1]) == int(hidden_dim):
                cached = weight.detach().to(device=device, dtype=torch.float32)
                self._classifier_cache[cache_key] = cached
                return cached
        raise ValueError(f"No frozen language head matches image feature dim {hidden_dim}")

    def _input_embedding_weight_for(self, device: torch.device, hidden_dim: int) -> torch.Tensor:
        cache_key = (str(device), int(hidden_dim))
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        candidates = []
        input_embeddings = self.model.get_input_embeddings()
        output_embeddings = self.model.get_output_embeddings()
        if input_embeddings is not None and hasattr(input_embeddings, "weight"):
            candidates.append(input_embeddings.weight)
        if output_embeddings is not None and hasattr(output_embeddings, "weight"):
            candidates.append(output_embeddings.weight)
        for weight in candidates:
            if int(weight.shape[-1]) == int(hidden_dim):
                cached = weight.detach().to(device=device, dtype=torch.float32)
                self._embedding_cache[cache_key] = cached
                return cached
        raise ValueError(f"No frozen token embedding matches image feature dim {hidden_dim}")

    def _get_image_feature_model(self) -> Any:
        if hasattr(self.model, "get_image_features"):
            return self.model
        inner_model = getattr(self.model, "model", None)
        if inner_model is not None and hasattr(inner_model, "get_image_features"):
            return inner_model
        raise ValueError("Loaded LLaVA model does not expose get_image_features")

    def _image_feature_kwargs(self, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
        if "pixel_values" not in inputs:
            raise ValueError("LLaVA inputs do not contain pixel_values")
        kwargs: dict[str, Any] = {"pixel_values": inputs["pixel_values"]}
        config = getattr(self.model, "config", None)
        if config is not None:
            if hasattr(config, "vision_feature_layer"):
                kwargs["vision_feature_layer"] = getattr(config, "vision_feature_layer")
            if hasattr(config, "vision_feature_select_strategy"):
                kwargs["vision_feature_select_strategy"] = getattr(
                    config,
                    "vision_feature_select_strategy",
                )
        if "image_sizes" in inputs:
            kwargs["image_sizes"] = inputs["image_sizes"]
        return kwargs

    def _precompute_dynamic_image_features(
        self,
        inputs: dict[str, torch.Tensor],
        method: str,
        question: str,
        category: str,
    ) -> Any:
        feature_model = self._get_image_feature_model()
        with torch.inference_mode():
            original_features = feature_model.get_image_features(**self._image_feature_kwargs(inputs))
        return self._reduce_image_features(original_features, method, question, category)

    @contextmanager
    def _precomputed_image_features_context(self, reduced_features: Any):
        patched = []
        candidates = [self.model]
        inner_model = getattr(self.model, "model", None)
        if inner_model is not None:
            candidates.append(inner_model)

        def wrapped_get_image_features(*args, **kwargs):
            del args, kwargs
            return reduced_features

        for candidate in candidates:
            if not hasattr(candidate, "get_image_features"):
                continue
            original = candidate.get_image_features
            candidate.get_image_features = wrapped_get_image_features
            patched.append((candidate, original))
        try:
            yield
        finally:
            for candidate, original in patched:
                candidate.get_image_features = original

    def _reduce_image_features(self, features: Any, method: str, question: str, category: str) -> Any:
        if torch.is_tensor(features):
            if features.ndim in {2, 3}:
                original_tokens = int(features.shape[-2])
                base_hypothesis_count = 0
                if method == "k_semreduce":
                    classifier = self._semantic_classifier_for(features.device, int(features.shape[-1]))
                    result = k_semreduce(
                        patch_tokens=features,
                        classifier=classifier,
                        config=self.k_config,
                    )
                    hypotheses: list[str] = []
                    requested = self.k_config.num_semantic_classes
                elif method == "answer_adaptive_k_semreduce":
                    semantic_head, hypotheses, base_hypothesis_count = self._answer_adaptive_semantic_head(
                        features,
                        question,
                        category,
                    )
                    result = answer_adaptive_k_semreduce(
                        patch_tokens=features,
                        hypothesis_embeddings=semantic_head,
                        config=self.adaptive_config,
                    )
                    requested = len(hypotheses)
                else:
                    raise ValueError(f"Unsupported reduction method: {method}")

                selected_indices = result.selected_classes
                if selected_indices.ndim > 1:
                    selected_indices = selected_indices[0]
                selected = [
                    hypotheses[int(index)]
                    for index in selected_indices.tolist()
                    if hypotheses and int(index) < len(hypotheses)
                ]
                self._last_reduction_meta = {
                    "original_image_tokens": original_tokens,
                    "reduced_image_tokens": int(result.patch_tokens.shape[-2]),
                    "base_hypothesis_count": int(base_hypothesis_count)
                    if method == "answer_adaptive_k_semreduce"
                    else 0,
                    "hypothesis_multiplier": int(self.adaptive_hypothesis_multiplier)
                    if method == "answer_adaptive_k_semreduce"
                    else 0,
                    "requested_hypothesis_count": int(requested),
                    "selected_hypothesis_count": int(result.patch_tokens.shape[-2]),
                    "hypotheses": hypotheses,
                    "selected_hypotheses": selected,
                    "mean_region_mass": float(result.masses.float().mean().item()),
                    "max_region_mass": int(result.masses.max().item()),
                    "min_region_mass": int(result.masses.min().item()),
                    "dynamic_token_count": True,
                    "reduction_stage": "image_feature_level_after_vision_tower_and_projector",
                }
                return result.patch_tokens
            return features
        if hasattr(features, "pooler_output"):
            features.pooler_output = self._reduce_image_features(features.pooler_output, method, question, category)
            return features
        if isinstance(features, list):
            return [self._reduce_image_features(item, method, question, category) for item in features]
        if isinstance(features, tuple):
            return tuple(self._reduce_image_features(item, method, question, category) for item in features)
        return features

    def _answer_adaptive_semantic_head(
        self,
        features: torch.Tensor,
        question: str,
        category: str,
    ) -> tuple[torch.Tensor, list[str], int]:
        hidden_dim = int(features.shape[-1])
        device = features.device

        base = extract_question_hypotheses(question, category)
        hypotheses = expand_hypotheses(base, self.adaptive_hypothesis_multiplier)
        embeddings = torch.stack(
            [self._phrase_embedding(label, device, hidden_dim) for label in hypotheses],
            dim=0,
        )
        return embeddings, hypotheses, len(dedupe_preserve_order(base))

    def _top_visual_vocabulary(
        self,
        global_token: torch.Tensor,
        device: torch.device,
        hidden_dim: int,
    ) -> list[str]:
        labels, embeddings = self._semantic_vocabulary_embeddings(device, hidden_dim)
        norm_global = F.layer_norm(global_token.float().unsqueeze(0), global_token.shape[-1:]).squeeze(0)
        scores = norm_global @ embeddings.T
        order = torch.argsort(scores, descending=True)
        return [labels[int(index)] for index in order.tolist()]

    def _semantic_vocabulary_embeddings(
        self,
        device: torch.device,
        hidden_dim: int,
    ) -> tuple[list[str], torch.Tensor]:
        cache_key = (str(device), int(hidden_dim))
        if cache_key in self._semantic_vocab_cache:
            return self._semantic_vocab_cache[cache_key]
        labels = dedupe_preserve_order(SEMANTIC_VOCABULARY + COLORS + KNOWN_VISUAL_PHRASES)
        embeddings = torch.stack(
            [self._phrase_embedding(label, device, hidden_dim) for label in labels],
            dim=0,
        )
        self._semantic_vocab_cache[cache_key] = (labels, embeddings)
        return labels, embeddings

    def _phrase_embedding(self, phrase: str, device: torch.device, hidden_dim: int) -> torch.Tensor:
        weight = self._input_embedding_weight_for(device, hidden_dim)
        ids = self.tokenizer.encode(" " + phrase, add_special_tokens=False)
        ids = [token_id for token_id in ids if 0 <= int(token_id) < int(weight.shape[0])]
        if not ids:
            ids = self.tokenizer.encode(phrase, add_special_tokens=False)
            ids = [token_id for token_id in ids if 0 <= int(token_id) < int(weight.shape[0])]
        if not ids:
            vector = torch.zeros(hidden_dim, device=device, dtype=torch.float32)
            vector[0] = 1.0
            return vector
        index = torch.tensor(ids, dtype=torch.long, device=device)
        vector = weight[index].mean(dim=0)
        return F.normalize(vector.float(), p=2, dim=0)


def load_mme_examples(
    dataset_name: str,
    split: str,
    limit_images: int | None,
    seed: int,
    sampling: str,
    category: str,
) -> list[MMEExample]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    grouped: OrderedDict[str, list[tuple[int, dict[str, Any]]]] = OrderedDict()
    for index, row in enumerate(dataset):
        row_category = str(row["category"])
        if category and row_category.lower() != category.lower():
            continue
        image_id = str(row["question_id"])
        grouped.setdefault(image_id, []).append((index, row))

    selected_ids = select_image_ids(grouped, limit_images, seed, sampling)
    examples = []
    for image_id in selected_ids:
        rows = sorted(grouped[image_id], key=lambda item: item[0])
        for _, row in rows:
            examples.append(
                MMEExample(
                    image_id=image_id,
                    question_id=str(row["question_id"]),
                    question=str(row["question"]),
                    label=normalize_label(row["answer"]),
                    image=row["image"],
                    category=str(row["category"]),
                    dimension=category_dimension(str(row["category"])),
                )
            )
    return examples


def select_image_ids(
    grouped: OrderedDict[str, list[tuple[int, dict[str, Any]]]],
    limit_images: int | None,
    seed: int,
    sampling: str,
) -> list[str]:
    ids = list(grouped.keys())
    if limit_images is None or limit_images >= len(ids):
        return ids
    rng = random.Random(seed)
    if sampling == "first":
        return ids[:limit_images]
    if sampling == "shuffle":
        shuffled = ids[:]
        rng.shuffle(shuffled)
        return shuffled[:limit_images]

    by_category: dict[str, list[str]] = defaultdict(list)
    for image_id in ids:
        row = grouped[image_id][0][1]
        by_category[str(row["category"])].append(image_id)
    for values in by_category.values():
        rng.shuffle(values)

    selected = []
    categories = sorted(by_category)
    while len(selected) < limit_images and any(by_category.values()):
        for cat in categories:
            if by_category[cat]:
                selected.append(by_category[cat].pop(0))
                if len(selected) >= limit_images:
                    break
    return selected


def extract_question_hypotheses(question: str, category: str) -> list[str]:
    text = question.lower()
    cleaned = re.sub(r"please answer yes or no\.?", " ", text)
    concepts: list[str] = []
    concepts.extend(CATEGORY_HINTS.get(category, []))
    concepts.extend(CATEGORY_HINTS.get(category.lower(), []))

    for phrase in KNOWN_VISUAL_PHRASES:
        if phrase in cleaned:
            concepts.append(phrase)
    for phrase in RELATION_PHRASES:
        if phrase in cleaned:
            concepts.append(f"{phrase} relation")
    for color in COLORS:
        if re.search(rf"\b{re.escape(color)}\b", cleaned):
            concepts.append(color)
            concepts.append(f"{color} object")

    for quoted in re.findall(r"'([^']+)'|\"([^\"]+)\"", question):
        value = next((part for part in quoted if part), "").strip().lower()
        if value and value not in {"yes", "no"}:
            concepts.append(value[:48])

    for token in re.findall(r"[a-z0-9]+", cleaned):
        if token in STOPWORDS:
            continue
        if len(token) < 3 and not token.isdigit():
            continue
        concepts.append(singularize(token))
    if not concepts:
        concepts.extend(["object", "scene", "visual evidence"])
    return dedupe_preserve_order(concepts)


def expand_hypotheses(base: list[str], multiplier: int) -> list[str]:
    base_hypotheses = dedupe_preserve_order(base)
    if not base_hypotheses:
        base_hypotheses = ["object", "scene", "visual evidence"]

    multiplier = max(1, int(multiplier))
    templates = [
        "{label}",
        "visual evidence of {label}",
        "detailed {label} region",
    ]
    while len(templates) < multiplier:
        templates.append(f"supporting visual cue {len(templates) + 1} for {{label}}")

    expanded: list[str] = []
    for template in templates[:multiplier]:
        for label in base_hypotheses:
            expanded.append(template.format(label=label))
    return expanded


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        normalized = " ".join(str(value).strip().lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def category_dimension(category: str) -> str:
    if category in PERCEPTION_CATEGORIES:
        return "perception"
    if category in COGNITION_CATEGORIES:
        return "cognition"
    return "unknown"


def summarize_mme(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": compute_mme_metrics(records),
        "dimensions": {
            dimension: compute_mme_metrics([record for record in records if record["dimension"] == dimension])
            for dimension in sorted({record["dimension"] for record in records})
        },
        "categories": {
            category: compute_mme_metrics([record for record in records if record["category"] == category])
            for category in sorted({record["category"] for record in records})
        },
    }


def compute_mme_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(records)
    if total == 0:
        return empty_metrics()
    labels = [record["label"] for record in records]
    preds = [record["prediction"] for record in records]
    tp = sum(1 for label, pred in zip(labels, preds) if label == "yes" and pred == "yes")
    tn = sum(1 for label, pred in zip(labels, preds) if label == "no" and pred == "no")
    fp = sum(1 for label, pred in zip(labels, preds) if label == "no" and pred == "yes")
    fn = sum(1 for label, pred in zip(labels, preds) if label == "yes" and pred == "no")
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_image[str(record["image_id"])].append(record)
    image_total = len(by_image)
    paired_accuracy = (
        sum(1 for image_records in by_image.values() if all(record["correct"] for record in image_records))
        / image_total
        if image_total
        else 0.0
    )
    return {
        "total_questions": total,
        "total_images": image_total,
        "accuracy": accuracy,
        "paired_accuracy": paired_accuracy,
        "mme_score": 100.0 * accuracy + 100.0 * paired_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": sum(1 for pred in preds if pred == "yes") / total,
    }


def empty_metrics() -> dict[str, float | int]:
    return {
        "total_questions": 0,
        "total_images": 0,
        "accuracy": 0.0,
        "paired_accuracy": 0.0,
        "mme_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "yes_ratio": 0.0,
    }


def delta_overall(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = ["accuracy", "paired_accuracy", "mme_score", "precision", "recall", "f1", "yes_ratio"]
    return {key: float(current[key]) - float(baseline[key]) for key in keys}


def write_metric_csvs(output_dir: Path, summary: dict[str, Any], methods: list[str]) -> None:
    with (output_dir / "category_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "category", "questions", "images", "accuracy", "paired_accuracy", "mme_score"])
        for method in methods:
            for category, metric in summary[method]["categories"].items():
                writer.writerow(metric_row(method, category, metric))

    with (output_dir / "dimension_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "dimension", "questions", "images", "accuracy", "paired_accuracy", "mme_score"])
        for method in methods:
            for dimension, metric in summary[method]["dimensions"].items():
                writer.writerow(metric_row(method, dimension, metric))


def metric_row(method: str, name: str, metric: dict[str, Any]) -> list[Any]:
    return [
        method,
        name,
        metric["total_questions"],
        metric["total_images"],
        metric["accuracy"],
        metric["paired_accuracy"],
        metric["mme_score"],
    ]


def format_overall_table(summary: dict[str, Any], methods: list[str]) -> str:
    lines = [
        "method                          q      img    acc     acc+    mme     yes",
        "------------------------------  -----  -----  ------  ------  ------  ------",
    ]
    for method in methods:
        metric = summary[method]["overall"]
        lines.append(
            f"{method:<30}  {int(metric['total_questions']):>5}  "
            f"{int(metric['total_images']):>5}  {metric['accuracy']:>6.3f}  "
            f"{metric['paired_accuracy']:>6.3f}  {metric['mme_score']:>6.1f}  "
            f"{metric['yes_ratio']:>6.3f}"
        )
    return "\n".join(lines)


def format_dimension_table(summary: dict[str, Any], methods: list[str]) -> str:
    lines = [
        "method                          dimension   q      img    acc     acc+    mme",
        "------------------------------  ----------  -----  -----  ------  ------  ------",
    ]
    for method in methods:
        for dimension, metric in summary[method]["dimensions"].items():
            lines.append(
                f"{method:<30}  {dimension:<10}  {int(metric['total_questions']):>5}  "
                f"{int(metric['total_images']):>5}  {metric['accuracy']:>6.3f}  "
                f"{metric['paired_accuracy']:>6.3f}  {metric['mme_score']:>6.1f}"
            )
    return "\n".join(lines)


def build_prompt(processor: Any, question: str) -> str:
    instruction = f"{question}\n\nAnswer exactly one word: yes or no. Do not add explanation."
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"USER: <image>\n{instruction}\nASSISTANT:"


def force_image_placeholder_count(
    inputs: Any,
    image_token_id: int,
    target_count: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    input_dict = dict(inputs)
    input_ids = input_dict["input_ids"]
    attention_mask = input_dict.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    rows = []
    masks = []
    for ids, mask in zip(input_ids, attention_mask):
        image_positions = torch.where(ids == image_token_id)[0]
        if int(image_positions.numel()) == 0:
            raise ValueError("The prompt contains no LLaVA image token placeholders")
        first = int(image_positions[0].item())
        last = int(image_positions[-1].item())
        before = ids[:first]
        after = ids[last + 1 :]
        before_mask = mask[:first]
        after_mask = mask[last + 1 :]
        image_ids = torch.full(
            (int(target_count),),
            int(image_token_id),
            dtype=ids.dtype,
            device=ids.device,
        )
        image_mask = torch.ones_like(image_ids)
        rows.append(torch.cat([before, image_ids, after], dim=0))
        masks.append(torch.cat([before_mask, image_mask, after_mask], dim=0))

    max_len = max(int(row.numel()) for row in rows)
    padded_rows = []
    padded_masks = []
    for row, mask in zip(rows, masks):
        pad_len = max_len - int(row.numel())
        if pad_len > 0:
            row = torch.cat(
                [
                    row,
                    torch.full((pad_len,), int(pad_token_id), dtype=row.dtype, device=row.device),
                ],
                dim=0,
            )
            mask = torch.cat(
                [mask, torch.zeros((pad_len,), dtype=mask.dtype, device=mask.device)],
                dim=0,
            )
        padded_rows.append(row)
        padded_masks.append(mask)

    input_dict["input_ids"] = torch.stack(padded_rows, dim=0)
    input_dict["attention_mask"] = torch.stack(padded_masks, dim=0)
    return input_dict


def image_feature_count(features: Any) -> int:
    if torch.is_tensor(features):
        if features.ndim == 2:
            return int(features.shape[0])
        if features.ndim == 3:
            if int(features.shape[0]) != 1:
                raise ValueError("Dynamic MME runner expects batch size 1")
            return int(features.shape[1])
        raise ValueError(f"Unsupported image feature tensor shape: {tuple(features.shape)}")
    if hasattr(features, "pooler_output"):
        return image_feature_count(features.pooler_output)
    if isinstance(features, (list, tuple)):
        return sum(image_feature_count(item) for item in features)
    raise ValueError(f"Cannot infer image feature count from {type(features)!r}")


def move_inputs_to_model(inputs: Any, model: Any) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    moved = {}
    for key, value in dict(inputs).items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@contextmanager
def nullcontext():
    yield


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def normalize_label(value: str) -> str:
    text = str(value).strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    raise ValueError(f"Expected yes/no label, got: {value!r}")


def normalize_yes_no(text: str) -> str:
    value = text.strip().lower()
    if value.startswith("yes") or " yes" in f" {value} ":
        return "yes"
    if value.startswith("no") or " no" in f" {value} ":
        return "no"
    return "no"


if __name__ == "__main__":
    main()
