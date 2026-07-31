"""Basic academic entity extraction via lexicon + metric regex.

Targets Concept types used by the Postgres/Neo4j schema:
``architecture``, ``dataset``, ``task``, ``metric``, ``method``, ``other``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedEntity:
    """A concept (or dataset/metric) mention found in text."""

    name: str
    concept_type: str
    aliases: tuple[str, ...] = ()
    metric_value: str | None = None


# Canonical lexicon: match phrase (lower) -> (display name, type, aliases).
# Longer phrases are preferred when spans overlap.
_LEXICON: dict[str, tuple[str, str, tuple[str, ...]]] = {
    # Architectures
    "spiking transformer": ("Spiking Transformer", "architecture", ()),
    "vision transformer": ("Vision Transformer", "architecture", ("ViT",)),
    "transformer": ("Transformer", "architecture", ()),
    "bert": ("BERT", "architecture", ()),
    "gpt": ("GPT", "architecture", ()),
    "resnet": ("ResNet", "architecture", ()),
    "lstm": ("LSTM", "architecture", ()),
    "gru": ("GRU", "architecture", ()),
    "cnn": ("CNN", "architecture", ("convolutional neural network",)),
    "mlp": ("MLP", "architecture", ("multi-layer perceptron",)),
    "spiking neural network": ("Spiking Neural Network", "architecture", ("SNN",)),
    "snn": ("SNN", "architecture", ("spiking neural network",)),
    # Methods
    "self-attention": ("Self-Attention", "method", ("self attention",)),
    "self attention": ("Self-Attention", "method", ("self-attention",)),
    "multi-head attention": ("Multi-Head Attention", "method", ()),
    "cross-attention": ("Cross-Attention", "method", ("cross attention",)),
    "layer normalization": ("Layer Normalization", "method", ("layer norm",)),
    "layer norm": ("Layer Normalization", "method", ("layer normalization",)),
    "batch normalization": ("Batch Normalization", "method", ("batch norm",)),
    "dropout": ("Dropout", "method", ()),
    "backpropagation": ("Backpropagation", "method", ("backprop",)),
    "gradient descent": ("Gradient Descent", "method", ()),
    "adam": ("Adam", "method", ()),
    "adamw": ("AdamW", "method", ()),
    "lora": ("LoRA", "method", ()),
    "retrieval augmented generation": (
        "Retrieval-Augmented Generation",
        "method",
        ("RAG",),
    ),
    "rag": ("RAG", "method", ("retrieval augmented generation",)),
    "beam search": ("Beam Search", "method", ()),
    "knowledge distillation": ("Knowledge Distillation", "method", ()),
    "contrastive learning": ("Contrastive Learning", "method", ()),
    "fine-tuning": ("Fine-Tuning", "method", ("finetuning", "fine tuning")),
    "finetuning": ("Fine-Tuning", "method", ("fine-tuning",)),
    "pretraining": ("Pretraining", "method", ("pre-training",)),
    "pre-training": ("Pretraining", "method", ("pretraining",)),
    "leiden": ("Leiden", "method", ()),
    "community detection": ("Community Detection", "method", ()),
    # Datasets
    "imagenet": ("ImageNet", "dataset", ()),
    "cifar-10": ("CIFAR-10", "dataset", ("CIFAR10", "cifar10")),
    "cifar10": ("CIFAR-10", "dataset", ("CIFAR-10",)),
    "cifar-100": ("CIFAR-100", "dataset", ("CIFAR100", "cifar100")),
    "cifar100": ("CIFAR-100", "dataset", ("CIFAR-100",)),
    "mnist": ("MNIST", "dataset", ()),
    "fashion-mnist": ("Fashion-MNIST", "dataset", ()),
    "coco": ("COCO", "dataset", ("MS COCO",)),
    "squad": ("SQuAD", "dataset", ()),
    "glue": ("GLUE", "dataset", ()),
    "wikitext": ("WikiText", "dataset", ()),
    "shakespeare": ("Shakespeare", "dataset", ()),
    # Tasks
    "language modeling": ("Language Modeling", "task", ()),
    "machine translation": ("Machine Translation", "task", ()),
    "named entity recognition": (
        "Named Entity Recognition",
        "task",
        ("NER",),
    ),
    "ner": ("Named Entity Recognition", "task", ("named entity recognition",)),
    "question answering": ("Question Answering", "task", ("QA",)),
    "image classification": ("Image Classification", "task", ()),
    "object detection": ("Object Detection", "task", ()),
    "semantic segmentation": ("Semantic Segmentation", "task", ()),
    "text classification": ("Text Classification", "task", ()),
    "speech recognition": ("Speech Recognition", "task", ()),
    # Metrics (name only; values via regex)
    "accuracy": ("Accuracy", "metric", ()),
    "top-1 accuracy": ("Top-1 Accuracy", "metric", ("top1 accuracy", "top 1")),
    "top-5 accuracy": ("Top-5 Accuracy", "metric", ("top5 accuracy", "top 5")),
    "f1": ("F1", "metric", ("f1-score", "f1 score")),
    "f1-score": ("F1", "metric", ("f1",)),
    "bleu": ("BLEU", "metric", ()),
    "rouge": ("ROUGE", "metric", ()),
    "rouge-l": ("ROUGE-L", "metric", ()),
    "perplexity": ("Perplexity", "metric", ()),
    "map": ("mAP", "metric", ("mean average precision",)),
    "auc": ("AUC", "metric", ()),
    "precision": ("Precision", "metric", ()),
    "recall": ("Recall", "metric", ()),
}

_METRIC_VALUE_RE = re.compile(
    r"\b(?P<name>"
    r"accuracy|top-?\s?1(?:\s+accuracy)?|top-?\s?5(?:\s+accuracy)?"
    r"|f1(?:-?\s?score)?|bleu|rouge-?l?|perplexity|mAP|auc|precision|recall"
    r")\b"
    r"(?:\s*(?:score|of|is|=|:))?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)

_METRIC_NAME_MAP = {
    "accuracy": ("Accuracy", "metric"),
    "top-1": ("Top-1 Accuracy", "metric"),
    "top1": ("Top-1 Accuracy", "metric"),
    "top 1": ("Top-1 Accuracy", "metric"),
    "top-1 accuracy": ("Top-1 Accuracy", "metric"),
    "top1 accuracy": ("Top-1 Accuracy", "metric"),
    "top 1 accuracy": ("Top-1 Accuracy", "metric"),
    "top-5": ("Top-5 Accuracy", "metric"),
    "top5": ("Top-5 Accuracy", "metric"),
    "top 5": ("Top-5 Accuracy", "metric"),
    "top-5 accuracy": ("Top-5 Accuracy", "metric"),
    "top5 accuracy": ("Top-5 Accuracy", "metric"),
    "top 5 accuracy": ("Top-5 Accuracy", "metric"),
    "f1": ("F1", "metric"),
    "f1-score": ("F1", "metric"),
    "f1 score": ("F1", "metric"),
    "bleu": ("BLEU", "metric"),
    "rouge": ("ROUGE", "metric"),
    "rouge-l": ("ROUGE-L", "metric"),
    "rougel": ("ROUGE-L", "metric"),
    "perplexity": ("Perplexity", "metric"),
    "map": ("mAP", "metric"),
    "auc": ("AUC", "metric"),
    "precision": ("Precision", "metric"),
    "recall": ("Recall", "metric"),
}


def _compile_lexicon_patterns() -> list[tuple[re.Pattern[str], str, str, tuple[str, ...]]]:
    items = sorted(_LEXICON.items(), key=lambda kv: len(kv[0]), reverse=True)
    compiled: list[tuple[re.Pattern[str], str, str, tuple[str, ...]]] = []
    for phrase, (name, ctype, aliases) in items:
        # Allow flexible whitespace / hyphenation for multi-word phrases.
        escaped = re.escape(phrase).replace(r"\ ", r"[\s\-]+")
        pat = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        compiled.append((pat, name, ctype, aliases))
    return compiled


_LEXICON_PATTERNS = _compile_lexicon_patterns()


def extract_entities(text: str) -> list[ExtractedEntity]:
    """Extract concept mentions from a single text span.

    Uses a curated academic ML lexicon plus regex for metric values.
    Overlapping lexicon matches keep the longest phrase.
    """
    if not text or not text.strip():
        return []

    occupied: list[tuple[int, int]] = []
    found: dict[tuple[str, str], ExtractedEntity] = {}

    def _overlaps(start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in occupied)

    for pat, name, ctype, aliases in _LEXICON_PATTERNS:
        for match in pat.finditer(text):
            start, end = match.span()
            if _overlaps(start, end):
                continue
            occupied.append((start, end))
            key = (name.lower(), ctype)
            if key not in found:
                found[key] = ExtractedEntity(name=name, concept_type=ctype, aliases=aliases)

    for match in _METRIC_VALUE_RE.finditer(text):
        raw_name = re.sub(r"\s+", " ", match.group("name").strip().lower())
        raw_name = raw_name.replace("top- 1", "top-1").replace("top- 5", "top-5")
        mapped = _METRIC_NAME_MAP.get(raw_name)
        if mapped is None:
            continue
        name, ctype = mapped
        value = match.group("value")
        key = (name.lower(), ctype)
        existing = found.get(key)
        if existing is None or (existing.metric_value is None and value):
            found[key] = ExtractedEntity(
                name=name,
                concept_type=ctype,
                aliases=(),
                metric_value=value,
            )

    return sorted(found.values(), key=lambda e: (e.concept_type, e.name.lower()))


def extract_entities_from_chunks(
    chunks: list[str],
) -> list[list[ExtractedEntity]]:
    """Run ``extract_entities`` on each chunk independently."""
    return [extract_entities(chunk) for chunk in chunks]
