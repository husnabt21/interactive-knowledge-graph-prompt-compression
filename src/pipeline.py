"""End-to-end orchestration shared by the Day 3 UI, demo script and tests.

Thin wiring only: calls the existing Day 1 and Day 2 public APIs in order.
No analysis logic lives here, so results are identical to running the steps
by hand (as scripts/day2_demo.py does).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx

from src.compression.compresser import (
    CompressionConfig,
    CompressionError,
    CompressionResult,
    compress_graph,
)
from src.compression.protection_rules import ProtectionRuleError
from src.embeddings.embedder import EmbeddingError, EmbeddingResult, embed_units
from src.evaluation.baseline import BaselineError, TokenCounter
from src.evaluation.metrics import EvaluationError, EvaluationResult, evaluate_compression
from src.extraction.unit_classifier import (
    ClassificationError,
    ClassifiedUnit,
    classify_units,
)
from src.graph.builder import GraphBuildError, GraphConfig, build_semantic_graph
from src.graph.importance import ImportanceError, annotate_graph, compute_importance
from src.graph.metrics import GraphMetricsError, compute_graph_metrics
from src.preprocessing.cleaner import EmptyPromptError, PromptCleaningError, clean_prompt
from src.preprocessing.segmenter import SegmentationError, segment_prompt

# Every typed error the backend raises on bad input or a failed stage.
# The UI and demo catch exactly these, so programming errors still surface.
PIPELINE_ERRORS: tuple[type[Exception], ...] = (
    EmptyPromptError,
    PromptCleaningError,
    SegmentationError,
    ClassificationError,
    EmbeddingError,
    GraphBuildError,
    ImportanceError,
    GraphMetricsError,
    ProtectionRuleError,
    CompressionError,
    BaselineError,
    EvaluationError,
)


@dataclass(frozen=True)
class AnalysisSettings:
    """User-facing knobs; defaults match the Day 1 / Day 2 defaults."""

    similarity_threshold: float = 0.30
    retention_ratio: float = 0.70
    protect_strong: bool = True
    remove_near_duplicates: bool = True


@dataclass(frozen=True, eq=False)
class AnalysisResult:
    """Everything the UI, demo and exports need from one run."""

    settings: AnalysisSettings
    cleaned_prompt: str
    classified: tuple[ClassifiedUnit, ...]
    embeddings: EmbeddingResult
    graph: nx.Graph                 # annotated Day 1 graph
    importance: Any                 # Day 1 ImportanceResult
    graph_metrics: Any              # Day 1 GraphMetrics
    compression: CompressionResult  # Day 2
    evaluation: EvaluationResult    # Day 2
    warnings: tuple[str, ...]


def run_analysis(
    prompt: str,
    settings: Optional[AnalysisSettings] = None,
    *,
    model: Any | None = None,
    counter: Optional[TokenCounter] = None,
    compute_semantic: bool = True,
) -> AnalysisResult:
    """Run the full pipeline on a prompt.

    Args:
        prompt: raw prompt text.
        settings: threshold / retention / protection / near-duplicate options.
        model: optional SentenceTransformer-like object (used for both the unit
            embeddings and the semantic-similarity metric). Tests pass a fake.
        counter: optional token counter (defaults to tiktoken cl100k_base).
        compute_semantic: set False to skip the semantic-similarity embedding.

    Raises:
        The typed errors listed in PIPELINE_ERRORS, and TypeError for a
        non-string prompt.
    """
    cfg = settings if settings is not None else AnalysisSettings()

    cleaned = clean_prompt(prompt)
    segmented = segment_prompt(cleaned)
    classified = classify_units(segmented)
    embeddings = embed_units(classified, model=model)
    graph = build_semantic_graph(
        classified, embeddings, GraphConfig(similarity_threshold=cfg.similarity_threshold)
    )
    importance = compute_importance(graph)
    annotate_graph(graph, importance)
    graph_metrics = compute_graph_metrics(graph)

    compression = compress_graph(
        graph,
        CompressionConfig(
            retention_ratio=cfg.retention_ratio,
            protect_strong=cfg.protect_strong,
            remove_near_duplicates=cfg.remove_near_duplicates,
        ),
        original_prompt=cleaned.text,
    )
    evaluation = evaluate_compression(
        compression, counter=counter, model=model, compute_semantic=compute_semantic
    )

    # evaluation.warnings already contains the compression warnings.
    warnings = _dedupe(
        [*cleaned.warnings, *segmented.warnings, *graph.graph["warnings"], *evaluation.warnings]
    )
    return AnalysisResult(
        settings=cfg,
        cleaned_prompt=cleaned.text,
        classified=tuple(classified),
        embeddings=embeddings,
        graph=graph,
        importance=importance,
        graph_metrics=graph_metrics,
        compression=compression,
        evaluation=evaluation,
        warnings=warnings,
    )


def _dedupe(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)