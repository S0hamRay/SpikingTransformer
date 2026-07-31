"""Tests for basic lexicon/regex entity extraction."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from db.entity_extraction import extract_entities, extract_entities_from_chunks
from db.sync import sync_paper_ingest


def test_extract_methods_and_architectures() -> None:
    text = (
        "We propose a Spiking Transformer with self-attention and LoRA "
        "fine-tuning for language modeling."
    )
    entities = extract_entities(text)
    by_type = {(e.name, e.concept_type) for e in entities}

    assert ("Spiking Transformer", "architecture") in by_type
    assert ("Self-Attention", "method") in by_type
    assert ("LoRA", "method") in by_type
    assert ("Language Modeling", "task") in by_type


def test_extract_datasets() -> None:
    entities = extract_entities("Trained on CIFAR-10 and ImageNet.")
    names = {e.name for e in entities if e.concept_type == "dataset"}
    assert names == {"CIFAR-10", "ImageNet"}


def test_extract_metric_with_value() -> None:
    entities = extract_entities("We achieve accuracy of 92.1% and BLEU 34.5.")
    metrics = {e.name: e.metric_value for e in entities if e.concept_type == "metric"}
    assert metrics.get("Accuracy") == "92.1"
    assert metrics.get("BLEU") == "34.5"


def test_longest_match_wins() -> None:
    entities = extract_entities("vision transformer models")
    names = [e.name for e in entities]
    assert "Vision Transformer" in names
    # Avoid double-counting the bare "transformer" inside the longer phrase.
    assert names.count("Transformer") == 0


def test_extract_from_chunks_independent() -> None:
    per_chunk = extract_entities_from_chunks(
        ["Uses Adam optimizer.", "Evaluated on MNIST."]
    )
    assert any(e.name == "Adam" for e in per_chunk[0])
    assert any(e.name == "MNIST" for e in per_chunk[1])


def test_empty_text() -> None:
    assert extract_entities("") == []
    assert extract_entities("   ") == []


def test_sync_links_extracted_entities() -> None:
    paper = {
        "filename": "demo.pdf",
        "title": "Demo Paper",
        "author": "Ada Lovelace",
        "text": "Abstract\nWe use Transformer and CIFAR-10. Accuracy of 90%.",
        "chunks": [
            SimpleNamespace(
                page_content="We use Transformer and CIFAR-10. Accuracy of 90%.",
                metadata={"position": 0},
            ),
        ],
    }

    pg = MagicMock()
    paper_id = "11111111-1111-1111-1111-111111111111"
    chunk_id = "33333333-3333-3333-3333-333333333333"
    concept_ids = {
        ("Transformer", "architecture"): "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        ("CIFAR-10", "dataset"): "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        ("Accuracy", "metric"): "cccccccc-cccc-cccc-cccc-cccccccccccc",
    }

    pg.upsert_paper.return_value = paper_id
    pg.upsert_author.return_value = "22222222-2222-2222-2222-222222222222"
    pg.replace_chunks.return_value = [chunk_id]

    def upsert_concept(*, name, concept_type, aliases=None):
        return concept_ids[(name, concept_type)]

    pg.upsert_concept.side_effect = upsert_concept
    pg.upsert_dataset.return_value = "dddddddd-dddd-dddd-dddd-dddddddddddd"

    graph = MagicMock()

    with patch("db.sync.sync_enabled", return_value=True):
        result = sync_paper_ingest(paper, postgres=pg, neo4j=graph)

    assert result["synced"] is True
    assert len(result["concept_ids"]) == 3

    pg.link_chunk_concept.assert_called()
    graph.link_mentions.assert_called()
    graph.link_uses_method.assert_called()
    graph.link_evaluates_on.assert_called()
    graph.link_reports_metric.assert_called()
    metric_kwargs = graph.link_reports_metric.call_args.kwargs
    assert metric_kwargs.get("value") == "90"
