"""Prompt cleaning: normalize raw prompt text without destroying its structure.

Input : raw prompt string
Output: CleanedPrompt (cleaned text + original text + warnings)

Design rules
------------
* Deterministic and rule-based (no LLM, no randomness).
* Preserve meaningful punctuation, line breaks, list indentation, and
  fenced code blocks, because the segmenter (Step 2) relies on them.
* Never fail silently: invalid input raises; borderline input is reported
  through ``CleanedPrompt.warnings``.
"""

from __future__ import annotations

import re
import textwrap
import unicodedata
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------
class PromptCleaningError(ValueError):
    """Raised when a prompt cannot be cleaned (e.g. too long)."""


class EmptyPromptError(PromptCleaningError):
    """Raised when the prompt is empty or contains no visible text."""


# --------------------------------------------------------------------------
# Configuration (centralized, immutable)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CleanerConfig:
    """Tunable cleaning parameters."""

    max_consecutive_blank_lines: int = 1   # blank lines kept between blocks
    tab_size: int = 4                      # tab width outside code fences
    normalize_quotes: bool = True          # curly quotes -> straight quotes
    min_words_warning: int = 3             # warn if cleaned prompt is shorter
    max_chars: int = 100_000               # reject absurdly large input


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CleanedPrompt:
    """Result of cleaning a prompt."""

    original: str
    text: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


# --------------------------------------------------------------------------
# Internal constants
# --------------------------------------------------------------------------
_QUOTE_MAP = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"',   # double quotes
    "\u2018": "'", "\u2019": "'", "\u201a": "'",   # single quotes / apostrophe
}

_SPACE_AND_INVISIBLE_MAP = {
    # unusual spaces -> normal space
    "\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u205f": " ", "\u3000": " ",
    **{chr(c): " " for c in range(0x2000, 0x200B)},
    # invisible characters -> removed
    "\u200b": None, "\u2060": None, "\ufeff": None,
    # vertical tab / form feed -> newline
    "\x0b": "\n", "\x0c": "\n",
}

_FENCE_MARKERS = ("```", "~~~")
_INLINE_SPACES = re.compile(r"(?<=\S) {2,}")  # runs of spaces AFTER text only,
                                              # so leading indentation survives


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def clean_prompt(text: str, config: CleanerConfig | None = None) -> CleanedPrompt:
    """Normalize a raw prompt while preserving its instruction structure.

    Args:
        text: Raw prompt text.
        config: Optional cleaning configuration.

    Returns:
        CleanedPrompt with the normalized text and any warnings.

    Raises:
        TypeError: if ``text`` is not a string.
        EmptyPromptError: if the prompt is empty / whitespace-only.
        PromptCleaningError: if the prompt exceeds ``config.max_chars``.
    """
    cfg = config or CleanerConfig()

    # ---- validation (fail loudly) ----
    if not isinstance(text, str):
        raise TypeError(f"Prompt must be a string, got {type(text).__name__}.")
    if len(text) > cfg.max_chars:
        raise PromptCleaningError(
            f"Prompt has {len(text)} characters; the limit is {cfg.max_chars}."
        )
    if not text.strip():
        raise EmptyPromptError("Prompt is empty or contains only whitespace.")

    warnings: list[str] = []

    # ---- character-level normalization ----
    s = unicodedata.normalize("NFC", text)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.translate(str.maketrans(_SPACE_AND_INVISIBLE_MAP))
    if cfg.normalize_quotes:
        s = s.translate(str.maketrans(_QUOTE_MAP))
    # drop remaining control characters (keep newline and tab)
    s = "".join(ch for ch in s if ch in "\n\t" or unicodedata.category(ch) != "Cc")

    # ---- line-level normalization (fence-aware) ----
    out: list[str] = []
    in_fence = False
    blank_run = 0

    for line in s.split("\n"):
        if line.strip().startswith(_FENCE_MARKERS):
            in_fence = not in_fence
            out.append(line.rstrip())
            blank_run = 0
            continue

        if in_fence:                       # code block: only strip trailing space
            out.append(line.rstrip())
            continue

        line = line.expandtabs(cfg.tab_size).rstrip()
        line = _INLINE_SPACES.sub(" ", line)

        if not line:
            blank_run += 1
            if blank_run > cfg.max_consecutive_blank_lines:
                continue                   # collapse extra blank lines
        else:
            blank_run = 0
        out.append(line)

    if in_fence:
        warnings.append("Unclosed code fence detected; content kept as-is.")

    cleaned = textwrap.dedent("\n".join(out)).strip()

    if not cleaned:
        raise EmptyPromptError("Prompt contains no visible text after cleaning.")

    if len(cleaned.split()) < cfg.min_words_warning:
        warnings.append(
            f"Very short prompt ({len(cleaned.split())} words); "
            "graph analysis may be uninformative."
        )

    return CleanedPrompt(original=text, text=cleaned, warnings=tuple(warnings))


# --------------------------------------------------------------------------
# Manual check:  python -m src.preprocessing.cleaner
# --------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "  You are a   senior data analyst.\r\n\r\n\r\n\r\n"
        "Task:   Summarize the report.\t\r\n"
        "- Use \u201cbullet points\u201d\n"
        "   - Keep it short  \n"
    )
    result = clean_prompt(sample)
    print("--- ORIGINAL (repr) ---")
    print(repr(sample))
    print("\n--- CLEANED ---")
    print(result.text)
    print(f"\nchars: {len(result.original)} -> {len(result.text)}")
    print(f"words: {result.word_count}")
    print(f"warnings: {list(result.warnings)}")