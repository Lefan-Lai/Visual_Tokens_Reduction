from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from early_semreduce import (
    FastFloodSemReduceConfig,
    FloodSemReduceConfig,
    KSemReduceConfig,
    SemReduceConfig,
    early_semreduce,
    fast_flood_semreduce,
    flood_semreduce,
    k_semreduce,
)


@dataclass(frozen=True)
class Example:
    question_id: str
    question: str
    label: str
    image: Any
    image_source: str
    category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare vanilla LLaVA 13B with semantic token reduction methods on POPE."
    )
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-13b-hf")
    parser.add_argument("--dataset-name", default="lmms-lab/POPE")
    parser.add_argument("--split", default="test")
    parser.add_argument("--category", default="adversarial")
    parser.add_argument("--methods", default="vanilla,fast_flood_semreduce")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/llava13b_early_semreduce_pope100")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prototype-tokens", type=int, default=128)
    parser.add_argument("--candidate-classes", type=int, default=64)
    parser.add_argument("--semantic-anchors", type=int, default=8)
    parser.add_argument("--cluster-iters", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-importance", type=float, default=0.25)
    parser.add_argument("--lambda-diversity", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--k-cluster-iters", type=int, default=3)
    parser.add_argument("--k-temperature", type=float, default=0.1)
    parser.add_argument("--k-lambda-importance", type=float, default=0.25)
    parser.add_argument("--k-gamma", type=float, default=1.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    known_methods = {"vanilla", "early_semreduce", "flood_semreduce", "fast_flood_semreduce", "k_semreduce"}
    unknown = [method for method in methods if method not in known_methods]
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(unknown)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SemanticReduce] Loading dataset {args.dataset_name}/{args.split}...")
    examples = load_pope_examples(args.dataset_name, args.split, args.category, args.limit, args.seed)
    print(f"[SemanticReduce] Loaded {len(examples)} examples.")

    print(f"[SemanticReduce] Loading model: {args.model_id}")
    runner = LlavaRunner(
        model_id=args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        load_in_4bit=args.load_in_4bit,
        prototype_tokens=args.prototype_tokens,
        candidate_classes=args.candidate_classes,
        semantic_anchors=args.semantic_anchors,
        cluster_iters=args.cluster_iters,
        temperature=args.temperature,
        lambda_importance=args.lambda_importance,
        lambda_diversity=args.lambda_diversity,
        gamma=args.gamma,
        k_cluster_iters=args.k_cluster_iters,
        k_temperature=args.k_temperature,
        k_lambda_importance=args.k_lambda_importance,
        k_gamma=args.k_gamma,
    )
    print("[SemanticReduce] Model ready. Running methods:", ", ".join(methods))

    labels: list[str] = []
    predictions: dict[str, list[str]] = {method: [] for method in methods}
    writers = {
        method: (output_dir / f"{method}.jsonl").open("w", encoding="utf-8")
        for method in methods
    }

    start = perf_counter()
    try:
        for index, example in enumerate(examples, start=1):
            labels.append(example.label)
            print(
                f"\n[{index}/{len(examples)}] qid={example.question_id} "
                f"label={example.label} question={example.question}"
            )
            for method in methods:
                method_start = perf_counter()
                answer = runner.ask_yes_no(
                    example.image,
                    example.question,
                    method=method,
                )
                elapsed = perf_counter() - method_start
                predictions[method].append(answer["prediction"])
                record = {
                    "question_id": example.question_id,
                    "image_source": example.image_source,
                    "category": example.category,
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
                writers[method].write(json.dumps(record, ensure_ascii=False) + "\n")
                writers[method].flush()
                print(
                    f"  - {method:<22} pred={record['prediction']:<3} "
                    f"conf={record['confidence']:.3f} correct={record['correct']} "
                    f"time={elapsed:.1f}s"
                )
    finally:
        for writer in writers.values():
            writer.close()

    metrics_by_method = {method: compute_metrics(labels, preds) for method, preds in predictions.items()}
    summary = dict(metrics_by_method)
    if "vanilla" in metrics_by_method:
        for method in methods:
            if method == "vanilla" or method not in metrics_by_method:
                continue
            summary[f"delta_{method}_minus_vanilla"] = {
                key: metrics_by_method[method][key] - metrics_by_method["vanilla"][key]
                for key in ["accuracy", "precision", "recall", "f1", "yes_ratio"]
            }
    summary["run"] = {
        "model_id": args.model_id,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "category": args.category,
        "limit": args.limit,
        "methods": methods,
        "prototype_tokens": args.prototype_tokens,
        "candidate_classes": args.candidate_classes,
        "semantic_anchors": args.semantic_anchors,
        "cluster_iters": args.cluster_iters,
        "temperature": args.temperature,
        "lambda_importance": args.lambda_importance,
        "lambda_diversity": args.lambda_diversity,
        "gamma": args.gamma,
        "k_cluster_iters": args.k_cluster_iters,
        "k_temperature": args.k_temperature,
        "k_lambda_importance": args.k_lambda_importance,
        "k_gamma": args.k_gamma,
        "load_in_4bit": args.load_in_4bit,
        "elapsed_sec": perf_counter() - start,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[SemanticReduce] Summary")
    print(format_metrics_table(metrics_by_method))
    print(f"\n[SemanticReduce] Wrote results to: {output_dir.resolve()}")


class LlavaRunner:
    def __init__(
        self,
        model_id: str,
        dtype: str,
        device_map: str,
        max_new_tokens: int,
        load_in_4bit: bool,
        prototype_tokens: int,
        candidate_classes: int,
        semantic_anchors: int,
        cluster_iters: int,
        temperature: float,
        lambda_importance: float,
        lambda_diversity: float,
        gamma: float,
        k_cluster_iters: int,
        k_temperature: float,
        k_lambda_importance: float,
        k_gamma: float,
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
        self.prototype_tokens = prototype_tokens
        self.semreduce_config = SemReduceConfig(
            num_prototypes=prototype_tokens,
            candidate_classes=candidate_classes,
            num_anchors=semantic_anchors,
            iterations=cluster_iters,
            temperature=temperature,
            lambda_importance=lambda_importance,
            lambda_diversity=lambda_diversity,
            gamma=gamma,
        )
        self.flood_config = FloodSemReduceConfig(
            candidate_classes=candidate_classes,
        )
        self.fast_flood_config = FastFloodSemReduceConfig(
            candidate_classes=candidate_classes,
        )
        self.k_config = KSemReduceConfig(
            num_semantic_classes=candidate_classes,
            iterations=k_cluster_iters,
            temperature=k_temperature,
            lambda_importance=k_lambda_importance,
            gamma=k_gamma,
        )
        self.image_token_id = self._image_token_id()
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        self.yes_token_ids = self._candidate_first_token_ids(["yes", "Yes", " yes", " Yes"])
        self.no_token_ids = self._candidate_first_token_ids(["no", "No", " no", " No"])
        self._classifier_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._last_reduction_meta: dict[str, Any] = {}

    def ask_yes_no(self, image: Any, question: str, method: str) -> dict:
        prompt = build_prompt(self.processor, question)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = move_inputs_to_model(inputs, self.model)
        self._last_reduction_meta = {}
        if method == "early_semreduce":
            inputs = force_image_placeholder_count(
                inputs,
                image_token_id=self.image_token_id,
                target_count=self.prototype_tokens,
                pad_token_id=int(self.pad_token_id),
            )
            context = self._semreduce_context("early_semreduce")
        elif method in {"flood_semreduce", "fast_flood_semreduce", "k_semreduce"}:
            reduced_features = self._precompute_dynamic_image_features(inputs, method)
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
        meta = {
            "algorithm": method,
            "yes_prob": confidence["yes"],
            "no_prob": confidence["no"],
        }
        if method == "early_semreduce":
            meta.update(
                {
                    "prototype_tokens": self.prototype_tokens,
                    "candidate_classes": self.semreduce_config.candidate_classes,
                    "semantic_anchors": self.semreduce_config.num_anchors,
                    "cluster_iters": self.semreduce_config.iterations,
                    "temperature": self.semreduce_config.temperature,
                }
            )
        elif method in {"flood_semreduce", "fast_flood_semreduce"}:
            config = self.fast_flood_config if method == "fast_flood_semreduce" else self.flood_config
            meta.update(
                {
                    "candidate_classes": config.candidate_classes,
                    "dynamic_token_count": True,
                    "reduction_stage": "image_feature_level_after_vision_tower_and_projector",
                    **self._last_reduction_meta,
                }
            )
        elif method == "k_semreduce":
            meta.update(
                {
                    "K": self.k_config.num_semantic_classes,
                    "candidate_classes": self.k_config.num_semantic_classes,
                    "cluster_iters": self.k_config.iterations,
                    "temperature": self.k_config.temperature,
                    "lambda_importance": self.k_config.lambda_importance,
                    "gamma": self.k_config.gamma,
                    "dynamic_token_count": False,
                    "reduction_stage": "image_feature_level_after_vision_tower_and_projector",
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

        shapes = [tuple(weight.shape) for weight in candidates]
        raise ValueError(
            f"No frozen LLaVA language head matches image feature dim {hidden_dim}; "
            f"candidate shapes: {shapes}"
        )

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

    def _precompute_dynamic_image_features(self, inputs: dict[str, torch.Tensor], method: str) -> Any:
        feature_model = self._get_image_feature_model()
        with torch.inference_mode():
            original_features = feature_model.get_image_features(**self._image_feature_kwargs(inputs))
        return self._reduce_image_features(original_features, method=method)

    @contextmanager
    def _semreduce_context(self, method: str):
        patched = []
        candidates = [self.model]
        inner_model = getattr(self.model, "model", None)
        if inner_model is not None:
            candidates.append(inner_model)

        def make_wrapped_get_image_features(original):
            def wrapped_get_image_features(*args, **kwargs):
                outputs = original(*args, **kwargs)
                return self._reduce_image_features(outputs, method=method)

            return wrapped_get_image_features

        for candidate in candidates:
            if not hasattr(candidate, "get_image_features"):
                continue
            original = candidate.get_image_features
            candidate.get_image_features = make_wrapped_get_image_features(original)
            patched.append((candidate, original))
        try:
            yield
        finally:
            for candidate, original in patched:
                candidate.get_image_features = original

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

    def _reduce_image_features(self, features: Any, method: str) -> Any:
        if torch.is_tensor(features):
            if features.ndim in {2, 3}:
                original_tokens = int(features.shape[-2])
                classifier = self._semantic_classifier_for(features.device, int(features.shape[-1]))
                if method == "early_semreduce":
                    result = early_semreduce(
                        patch_tokens=features,
                        classifier=classifier,
                        config=self.semreduce_config,
                    )
                elif method == "flood_semreduce":
                    result = flood_semreduce(
                        patch_tokens=features,
                        classifier=classifier,
                        config=self.flood_config,
                    )
                elif method == "fast_flood_semreduce":
                    result = fast_flood_semreduce(
                        patch_tokens=features,
                        classifier=classifier,
                        config=self.fast_flood_config,
                    )
                elif method == "k_semreduce":
                    result = k_semreduce(
                        patch_tokens=features,
                        classifier=classifier,
                        config=self.k_config,
                    )
                else:
                    raise ValueError(f"Unsupported reduction method: {method}")
                self._last_reduction_meta = {
                    "original_image_tokens": original_tokens,
                    "reduced_image_tokens": int(result.patch_tokens.shape[-2]),
                    "mean_region_mass": float(result.masses.float().mean().item()),
                    "max_region_mass": int(result.masses.max().item()),
                    "min_region_mass": int(result.masses.min().item()),
                }
                return result.patch_tokens
            return features
        if hasattr(features, "pooler_output"):
            features.pooler_output = self._reduce_image_features(features.pooler_output, method)
            return features
        if isinstance(features, list):
            return [self._reduce_image_features(item, method) for item in features]
        if isinstance(features, tuple):
            return tuple(self._reduce_image_features(item, method) for item in features)
        return features


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
                raise ValueError("Dynamic Flood-SemReduce LLaVA runner expects batch size 1")
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


def load_pope_examples(
    dataset_name: str,
    split: str,
    category: str,
    limit: int | None,
    seed: int,
) -> list[Example]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    if category:
        dataset = dataset.filter(lambda row: str(row.get("category", "")).lower() == category.lower())
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    if limit is not None:
        dataset = dataset.select(range(min(int(limit), len(dataset))))

    examples = []
    for row in dataset:
        examples.append(
            Example(
                question_id=str(row.get("question_id", row.get("id", len(examples)))),
                question=str(row["question"]),
                label=normalize_label(row["answer"]),
                image=row["image"],
                image_source=str(row.get("image_source", "")),
                category=str(row.get("category", category)),
            )
        )
    return examples


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


def compute_metrics(labels: list[str], predictions: list[str]) -> dict[str, float]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    total = len(labels)
    if total == 0:
        return {
            "total": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "yes_ratio": 0.0,
        }

    tp = sum(1 for label, pred in zip(labels, predictions) if label == "yes" and pred == "yes")
    tn = sum(1 for label, pred in zip(labels, predictions) if label == "no" and pred == "no")
    fp = sum(1 for label, pred in zip(labels, predictions) if label == "no" and pred == "yes")
    fn = sum(1 for label, pred in zip(labels, predictions) if label == "yes" and pred == "no")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total": total,
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": sum(1 for pred in predictions if pred == "yes") / total,
    }


def format_metrics_table(metrics_by_method: dict[str, dict[str, float]]) -> str:
    lines = [
        "method                  total  acc     precision  recall  f1      yes_ratio",
        "----------------------  -----  ------  ---------  ------  ------  ---------",
    ]
    for method, metric in metrics_by_method.items():
        lines.append(
            f"{method:<22}  {int(metric['total']):>5}  "
            f"{metric['accuracy']:>6.3f}  {metric['precision']:>9.3f}  "
            f"{metric['recall']:>6.3f}  {metric['f1']:>6.3f}  "
            f"{metric['yes_ratio']:>9.3f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
