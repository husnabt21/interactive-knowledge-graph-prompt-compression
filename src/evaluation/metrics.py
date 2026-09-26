"""Evaluation metrics for prompt compression (Day 2, Step 4).

Consumes a CompressionResult and returns an EvaluationResult.

Metric definitions
------------------
token metrics        : see baseline.py (tiktoken cl100k_base).
node_retention_percent = retained_nodes / original_nodes * 100
semantic_similarity  : cosine similarity between the embedding of the whole
                       original prompt and the whole compressed prompt.
                       This is a similarity score only; it does not guarantee
                       that an LLM would respond equivalently. The embedding
                       model reads only ~256 word pieces, so for long prompts
                       the score mostly reflects the start of each prompt.
instruction_fidelity : among protected units (policy strong types: instruction,
                       constraint, requirement, output_format), the fraction
                       PRESERVED, where preserved = retained, or removed as a
                       near-duplicate of a retained node. None if the prompt has
                       no protected units. It measures retention of protected
                       units, not model behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from src.compression.compresser import CompressionResult, Decision
from src.compression.protection_rules import DEFAULT_POLICY, ProtectionPolicy
from src.embeddings.embedder import EmbedderConfig, embed_texts
from src.evaluation.baseline import TokenCounter, compare_prompts

# Whitespace-word threshold above which the embedding model likely truncates.
MAX_RELIABLE_WORDS = 150


class EvaluationError(Exception):
    """Base class for evaluation errors."""


# --------------------------------------------------------------------------
# Individual metrics
# --------------------------------------------------------------------------
def node_retention_percent(retained_nodes: int, original_nodes: int) -> float:
    """retained / original * 100 (0.0 when original_nodes == 0)."""
    for name, value in (("retained_nodes", retained_nodes), ("original_nodes", original_nodes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if retained_nodes > original_nodes:
        raise ValueError("retained_nodes cannot exceed original_nodes")
    return (retained_nodes / original_nodes * 100.0) if original_nodes > 0 else 0.0


def semantic_similarity(
    original: str,
    compressed: str,
    embedding_config: Optional[EmbedderConfig] = None,
    model: Any | None = None,
) -> float:
    """Cosine similarity between whole-prompt embeddings, in [-1, 1].

    Pass ``model`` (any object with SentenceTransformer's ``encode``) to avoid
    loading the real model, e.g. in tests.
    """
    if not isinstance(original, str) or not isinstance(compressed, str):
        raise TypeError("original and compressed must be strings")
    if not original.strip() or not compressed.strip():
        raise EvaluationError("cannot compute semantic similarity for an empty prompt")

    vectors = np.asarray(
        embed_texts([original, compressed], embedding_config, model), dtype=np.float64
    )
    if vectors.ndim != 2 or vectors.shape[0] != 2:
        raise EvaluationError(f"unexpected embedding shape {vectors.shape}")
    a, b = vectors
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        raise EvaluationError("an embedding has zero norm; cosine similarity is undefined")
    return max(-1.0, min(1.0, float(np.dot(a, b) / (na * nb))))


@dataclass(frozen=True)
class InstructionFidelity:
    protected_total: int
    retained: int                    # protected units kept as nodes
    covered_by_duplicate: int        # removed, but a retained near-duplicate represents them
    lost_ids: tuple[str, ...]        # protected units neither retained nor represented
    fidelity: Optional[float]        # (retained + covered) / total; None if total == 0
    by_type: Mapping[str, Mapping[str, int]]

    @property
    def preserved(self) -> int:
        return self.retained + self.covered_by_duplicate


def compute_instruction_fidelity(
    result: CompressionResult,
    policy: ProtectionPolicy = DEFAULT_POLICY,
) -> InstructionFidelity:
    """Measure how many protected units survived compression."""
    _check_result(result)
    retained_ids = set(result.retained_node_ids)
    by_type: dict[str, dict[str, int]] = {}
    retained = covered = 0
    lost: list[str] = []

    for d in result.decisions:
        if not policy.is_protected(d.unit_type):   # raises InvalidUnitTypeError if unknown
            continue
        row = by_type.setdefault(
            d.unit_type, {"total": 0, "retained": 0, "covered_by_duplicate": 0, "lost": 0}
        )
        row["total"] += 1
        if d.decision is Decision.RETAIN:
            retained += 1
            row["retained"] += 1
        elif d.duplicate_of is not None and d.duplicate_of in retained_ids:
            covered += 1
            row["covered_by_duplicate"] += 1
        else:
            lost.append(d.node_id)
            row["lost"] += 1

    total = retained + covered + len(lost)
    return InstructionFidelity(
        protected_total=total,
        retained=retained,
        covered_by_duplicate=covered,
        lost_ids=tuple(lost),
        fidelity=((retained + covered) / total) if total else None,
        by_type=by_type,
    )


# --------------------------------------------------------------------------
# Combined result
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluationResult:
    # tokens
    original_tokens: int
    compressed_tokens: int
    token_reduction: int
    token_reduction_percent: float
    compression_ratio: Optional[float]
    # nodes
    original_nodes: int
    retained_nodes: int
    removed_nodes: int
    node_retention_percent: float
    requested_retention_ratio: float
    actual_retention_ratio: float
    # quality signals
    semantic_similarity: Optional[float]
    instruction_fidelity: Optional[float]
    protected_total: int
    protected_retained: int
    protected_covered_by_duplicate: int
    protected_lost_ids: tuple[str, ...]
    fidelity_by_type: Mapping[str, Mapping[str, int]]
    # provenance
    tokenizer: str
    tokenizer_exact: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "token_reduction": self.token_reduction,
            "token_reduction_percent": self.token_reduction_percent,
            "compression_ratio": self.compression_ratio,
            "original_nodes": self.original_nodes,
            "retained_nodes": self.retained_nodes,
            "removed_nodes": self.removed_nodes,
            "node_retention_percent": self.node_retention_percent,
            "requested_retention_ratio": self.requested_retention_ratio,
            "actual_retention_ratio": self.actual_retention_ratio,
            "semantic_similarity": self.semantic_similarity,
            "instruction_fidelity": self.instruction_fidelity,
            "protected_total": self.protected_total,
            "protected_retained": self.protected_retained,
            "protected_covered_by_duplicate": self.protected_covered_by_duplicate,
            "protected_lost_ids": list(self.protected_lost_ids),
            "fidelity_by_type": {k: dict(v) for k, v in self.fidelity_by_type.items()},
            "tokenizer": self.tokenizer,
            "tokenizer_exact": self.tokenizer_exact,
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        def row(label: str, value: str) -> str:
            return f"{label:<22}{value}"

        ratio = "n/a" if self.compression_ratio is None else f"{self.compression_ratio:.2f}x"
        sim = "n/a" if self.semantic_similarity is None else f"{self.semantic_similarity:.3f}"
        if self.instruction_fidelity is None:
            fid = "n/a (no protected units)"
        else:
            preserved = self.protected_retained + self.protected_covered_by_duplicate
            fid = (
                f"{self.instruction_fidelity:.2f} ({preserved}/{self.protected_total} protected "
                f"units preserved: {self.protected_retained} retained, "
                f"{self.protected_covered_by_duplicate} via near-duplicate representative)"
            )
            if self.protected_lost_ids:
                fid += f"; lost: {', '.join(self.protected_lost_ids)}"

        lines = [
            row("Original Nodes:", str(self.original_nodes)),
            row("Retained Nodes:", str(self.retained_nodes)),
            row("Removed Nodes:", str(self.removed_nodes)),
            row("Node Retention:", f"{self.node_retention_percent:.1f}%"),
            row("Original Tokens:", str(self.original_tokens)),
            row("Compressed Tokens:", str(self.compressed_tokens)),
            row("Token Reduction:", f"{self.token_reduction} ({self.token_reduction_percent:.1f}%)"),
            row("Compression Ratio:", ratio),
            row("Semantic Similarity:", sim),
            row("Instruction Fidelity:", fid),
            row("Tokenizer:", self.tokenizer),
        ]
        lines.extend(f"Warning: {w}" for w in self.warnings)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def evaluate_compression(
    result: CompressionResult,
    *,
    policy: ProtectionPolicy = DEFAULT_POLICY,
    counter: Optional[TokenCounter] = None,
    embedding_config: Optional[EmbedderConfig] = None,
    model: Any | None = None,
    compute_semantic: bool = True,
) -> EvaluationResult:
    """Evaluate a CompressionResult (tokens, nodes, similarity, fidelity).

    Set ``compute_semantic=False`` to skip the embedding step entirely.
    """
    _check_result(result)

    baseline = compare_prompts(result.original_prompt, result.compressed_prompt, counter)
    fidelity = compute_instruction_fidelity(result, policy)
    warnings: list[str] = list(baseline.warnings) + list(result.warnings)

    similarity: Optional[float] = None
    if compute_semantic:
        if result.original_prompt.strip() and result.compressed_prompt.strip():
            similarity = semantic_similarity(
                result.original_prompt, result.compressed_prompt, embedding_config, model
            )
            longest = max(len(result.original_prompt.split()), len(result.compressed_prompt.split()))
            if longest > MAX_RELIABLE_WORDS:
                warnings.append(
                    f"Prompt is long ({longest} words); the embedding model reads only about "
                    "256 word pieces, so the semantic similarity score mainly reflects the "
                    "start of each prompt."
                )
        else:
            warnings.append("Semantic similarity skipped: original or compressed prompt is empty.")

    if fidelity.fidelity is None:
        warnings.append("Instruction fidelity is undefined: the prompt has no protected units.")

    return EvaluationResult(
        original_tokens=baseline.original_tokens,
        compressed_tokens=baseline.compressed_tokens,
        token_reduction=baseline.token_reduction,
        token_reduction_percent=baseline.token_reduction_percent,
        compression_ratio=baseline.compression_ratio,
        original_nodes=result.original_node_count,
        retained_nodes=result.retained_node_count,
        removed_nodes=result.removed_node_count,
        node_retention_percent=node_retention_percent(
            result.retained_node_count, result.original_node_count
        ),
        requested_retention_ratio=result.requested_retention_ratio,
        actual_retention_ratio=result.actual_retention_ratio,
        semantic_similarity=similarity,
        instruction_fidelity=fidelity.fidelity,
        protected_total=fidelity.protected_total,
        protected_retained=fidelity.retained,
        protected_covered_by_duplicate=fidelity.covered_by_duplicate,
        protected_lost_ids=fidelity.lost_ids,
        fidelity_by_type=fidelity.by_type,
        tokenizer=baseline.tokenizer,
        tokenizer_exact=baseline.tokenizer_exact,
        warnings=tuple(warnings),
    )


def _check_result(result: Any) -> None:
    if not isinstance(result, CompressionResult):
        raise TypeError(f"expected CompressionResult, got {type(result).__name__}")


# --------------------------------------------------------------------------
# Manual check:  python -m src.evaluation.metrics
# Uses a hand-made graph, a whitespace token counter and a bag-of-words fake
# embedding model, so it needs no internet, tiktoken or sentence-transformers
# download.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import re

    import networkx as nx

    from src.compression.compresser import CompressionConfig, compress_graph

    class BagOfWordsModel:
        """Deterministic stand-in for SentenceTransformer (demo/tests only)."""

        def encode(self, items, normalize_embeddings=True, **kwargs):
            words = [re.findall(r"[a-z0-9]+", t.lower()) for t in items]
            vocab = sorted({w for ws in words for w in ws})
            col = {w: i for i, w in enumerate(vocab)}
            vecs = np.zeros((len(items), len(vocab)), dtype=np.float32)
            for r, ws in enumerate(words):
                for w in ws:
                    vecs[r, col[w]] += 1.0
            if normalize_embeddings:
                vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs

    g = nx.Graph(near_duplicate_pairs=[("N02", "N03", 0.95)])
    rows = [
        ("N01", 0, "You are a travel planner.", "role", 0.55, 1),
        ("N02", 1, "Plan a 3-day trip to Tokyo.", "instruction", 0.80, 3),
        ("N03", 2, "Plan a three day trip to Tokyo.", "instruction", 0.70, 4),
        ("N04", 3, "Tokyo has many museums.", "context", 0.40, 5),
        ("N05", 4, "Budget must not exceed $500.", "constraint", 0.85, 7),
        ("N06", 5, "Thanks a lot.", "other", 0.10, 9),
        ("N07", 6, "e.g. Visit Ueno Park.", "example", 0.35, 10),
    ]
    for nid, idx, text, utype, imp, line in rows:
        g.add_node(nid, node_id=nid, text=text, type=utype, importance=imp,
                   index=idx, kind="sentence", line_start=line)
    g.add_edge("N02", "N03", weight=0.95, near_duplicate=True)
    g.add_edge("N02", "N04", weight=0.40, near_duplicate=False)
    g.add_edge("N03", "N04", weight=0.38, near_duplicate=False)
    g.add_edge("N04", "N07", weight=0.50, near_duplicate=False)
    g.add_edge("N01", "N02", weight=0.35, near_duplicate=False)

    compression = compress_graph(g, CompressionConfig(retention_ratio=0.60))
    counter = TokenCounter("whitespace-demo", False, lambda t: len(t.split()))
    evaluation = evaluate_compression(compression, counter=counter, model=BagOfWordsModel())
    print(evaluation.summary())
    print(f"Protected lost: {list(evaluation.protected_lost_ids)}")