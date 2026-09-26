"""Sentence embeddings for information units.

Input : SegmentationResult / ClassifiedUnit(s) / InformationUnit(s) / str list
Output: EmbeddingResult (unit_ids, texts, vectors[n, dim], model_name)

Rules
-----
* Model: sentence-transformers 'all-MiniLM-L6-v2' (384 dims, runs locally).
* The model is loaded ONCE per process (lru_cache) and imported lazily.
* Vectors are L2-normalized, so cosine similarity == dot product.
* Row i of `vectors` always corresponds to unit_ids[i].
* Failures raise EmbeddingError; nothing is silently swallowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np

from src.preprocessing.segmenter import SegmentationResult

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------
class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced or are invalid."""


# --------------------------------------------------------------------------
# Configuration and result container
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbedderConfig:
    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = 32
    normalize: bool = True


@dataclass(frozen=True, eq=False)
class EmbeddingResult:
    """Embeddings for a list of units. Row i <-> unit_ids[i]."""

    unit_ids: tuple[str, ...]
    texts: tuple[str, ...]
    vectors: np.ndarray            # shape (n_units, dim), float32
    model_name: str

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.unit_ids):
            raise EmbeddingError(
                f"Vector matrix shape {self.vectors.shape} does not match "
                f"{len(self.unit_ids)} units."
            )

    def __len__(self) -> int:
        return len(self.unit_ids)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def vector(self, unit_id: str) -> np.ndarray:
        """Return the embedding of one unit by its ID."""
        try:
            return self.vectors[self.unit_ids.index(unit_id)]
        except ValueError:
            raise KeyError(f"Unknown unit id: {unit_id!r}") from None


# --------------------------------------------------------------------------
# Model loading (cached)
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def get_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Load (once) and return a SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        ) from exc
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise EmbeddingError(
            f"Could not load embedding model '{model_name}'. The first run "
            "needs internet access to download it; check the model name and "
            "your connection."
        ) from exc


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def embed_texts(
    texts: Sequence[str],
    config: EmbedderConfig | None = None,
    model: Any | None = None,
) -> np.ndarray:
    """Embed a list of strings into a (n, dim) float32 array.

    Args:
        texts: non-empty sequence of non-empty strings.
        config: optional embedder configuration.
        model: optional pre-loaded model (useful for tests).

    Raises:
        TypeError: if ``texts`` is a bare string or contains non-strings.
        EmbeddingError: empty input, empty text, model failure, or bad output.
    """
    cfg = config or EmbedderConfig()

    if isinstance(texts, str):
        raise TypeError("Expected a sequence of strings, got a single string.")
    items = list(texts)
    if not items:
        raise EmbeddingError("No texts to embed.")
    for i, t in enumerate(items):
        if not isinstance(t, str):
            raise TypeError(f"Text at position {i} is {type(t).__name__}, not str.")
        if not t.strip():
            raise EmbeddingError(f"Text at position {i} is empty.")

    model = model or get_model(cfg.model_name)
    try:
        vectors = model.encode(
            items,
            batch_size=cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=cfg.normalize,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(items):
        raise EmbeddingError(f"Unexpected embedding shape {vectors.shape}.")
    if not np.isfinite(vectors).all():
        raise EmbeddingError("Embeddings contain NaN or infinite values.")
    return vectors


def embed_units(
    units: SegmentationResult | Iterable[Any],
    config: EmbedderConfig | None = None,
    model: Any | None = None,
) -> EmbeddingResult:
    """Embed information units (anything with ``unit_id`` and ``text``).

    Accepts a SegmentationResult, or an iterable of InformationUnit /
    ClassifiedUnit objects.

    Raises:
        TypeError: an item lacks ``unit_id`` / ``text``.
        EmbeddingError: duplicate IDs or any failure in ``embed_texts``.
    """
    cfg = config or EmbedderConfig()
    items = units.units if isinstance(units, SegmentationResult) else tuple(units)

    ids: list[str] = []
    texts: list[str] = []
    for u in items:
        if not (hasattr(u, "unit_id") and hasattr(u, "text")):
            raise TypeError(
                f"Expected an object with 'unit_id' and 'text', got {type(u).__name__}."
            )
        ids.append(u.unit_id)
        texts.append(u.text)

    if len(set(ids)) != len(ids):
        raise EmbeddingError("Duplicate unit ids found; unit ids must be unique.")

    vectors = embed_texts(texts, cfg, model)
    return EmbeddingResult(
        unit_ids=tuple(ids),
        texts=tuple(texts),
        vectors=vectors,
        model_name=cfg.model_name,
    )


def similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity, shape (n, n), values clipped to [-1, 1]."""
    if vectors.ndim != 2:
        raise EmbeddingError(f"Expected a 2-D array, got shape {vectors.shape}.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise EmbeddingError("Cannot compute similarity for a zero vector.")
    unit = vectors / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)


# --------------------------------------------------------------------------
# Manual check:  python -m src.embeddings.embedder
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from src.extraction.unit_classifier import classify_units
    from src.preprocessing.cleaner import clean_prompt
    from src.preprocessing.segmenter import segment_prompt

    sample = (
        "You are an expert travel planner. Your goal is to help users plan trips.\n\n"
        "Task: Create a 3-day itinerary for Tokyo. Include e.g. museums and parks.\n\n"
        "Constraints:\n"
        "- Budget must not exceed $500.\n"
        "- Avoid crowded places.\n\n"
        "Output format:\n"
        "```json\n"
        '{"days": []}\n'
        "```\n"
    )
    classified = classify_units(segment_prompt(clean_prompt(sample)))

    first = embed_units(classified)
    second = embed_units(classified)          # model is reused, not reloaded
    sim = similarity_matrix(first.vectors)
    norms = np.linalg.norm(first.vectors, axis=1)

    print(f"Model: {first.model_name}")
    print(f"Units embedded: {len(first)}")
    print(f"Vector shape: {first.vectors.shape}")
    print(f"Vector norms (min/max): {norms.min():.3f} / {norms.max():.3f}")
    print(f"Deterministic (2 runs identical): {np.allclose(first.vectors, second.vectors)}")
    print(f"Similarity N01-N02 (role vs goal, related): {sim[0, 1]:.2f}")
    print(f"Similarity N05-N06 (both constraints):      {sim[4, 5]:.2f}")
    print(f"Similarity N01-N07 (role vs JSON block):    {sim[0, 6]:.2f}")