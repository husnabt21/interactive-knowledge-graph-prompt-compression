"""Token baseline for prompt compression (Day 2, Step 3).

Compares the original prompt with the compressed prompt using a fixed
tokenizer. No LLM is involved.

Definitions (also used by Step 5 metrics)::

    token_reduction         = original_tokens - compressed_tokens
    token_reduction_percent = token_reduction / original_tokens * 100
                              (0.0 when original_tokens == 0)
    compression_ratio       = original_tokens / compressed_tokens
                              (None when compressed_tokens == 0; ratio > 1 means
                               the compressed prompt is shorter)

A negative reduction is possible (for example when list markers are restored)
and is reported as-is.

Tokenizer: tiktoken ``cl100k_base``. If tiktoken or its encoding file is
unavailable (for example offline on first run) and ``allow_fallback`` is True,
a regex approximation is used and the result is labelled as not exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Optional

ENCODING_NAME = "cl100k_base"
_APPROX_PATTERN = re.compile(r"\w+|[^\w\s]")


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------
class BaselineError(Exception):
    """Base class for baseline errors."""


class TokenizerUnavailableError(BaselineError):
    """tiktoken or its encoding could not be loaded and no fallback is allowed."""


# --------------------------------------------------------------------------
# Token counter
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TokenCounter:
    """A named, deterministic token counter."""

    name: str
    exact: bool                                   # True only for real cl100k_base counts
    _count: Callable[[str], int] = field(repr=False, compare=False)

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if not text:
            return 0
        return int(self._count(text))


@lru_cache(maxsize=1)
def _load_encoding() -> Any:
    import tiktoken  # imported here so a missing package can be handled explicitly

    return tiktoken.get_encoding(ENCODING_NAME)


def _approx_count(text: str) -> int:
    return len(_APPROX_PATTERN.findall(text))


def get_token_counter(allow_fallback: bool = True) -> TokenCounter:
    """Return the cl100k_base counter, or an approximate one if allowed.

    Raises:
        TokenizerUnavailableError: tiktoken/encoding unavailable and
            allow_fallback is False.
    """
    try:
        encoding = _load_encoding()
    except (ImportError, OSError, ValueError) as exc:
        # ImportError: tiktoken not installed. OSError (includes requests
        # connection errors) / ValueError: encoding file could not be fetched or read.
        if not allow_fallback:
            raise TokenizerUnavailableError(
                f"could not load tiktoken encoding {ENCODING_NAME!r}: {exc}"
            ) from exc
        return TokenCounter("approx-regex (not tiktoken)", False, _approx_count)

    # disallowed_special=() so text containing e.g. "<|endoftext|>" is counted
    # as ordinary text instead of raising.
    return TokenCounter(
        f"{ENCODING_NAME} (tiktoken)",
        True,
        lambda t: len(encoding.encode(t, disallowed_special=())),
    )


def count_tokens(text: str, counter: Optional[TokenCounter] = None) -> int:
    """Count tokens in ``text`` (0 for the empty string)."""
    return (counter or get_token_counter()).count(text)


# --------------------------------------------------------------------------
# Comparison result
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BaselineComparison:
    original_tokens: int
    compressed_tokens: int
    token_reduction: int
    token_reduction_percent: float
    compression_ratio: Optional[float]
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
            "tokenizer": self.tokenizer,
            "tokenizer_exact": self.tokenizer_exact,
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        ratio = "n/a" if self.compression_ratio is None else f"{self.compression_ratio:.2f}x"
        lines = [
            f"Original Tokens:   {self.original_tokens}",
            f"Compressed Tokens: {self.compressed_tokens}",
            f"Token Reduction:   {self.token_reduction} ({self.token_reduction_percent:.1f}%)",
            f"Compression Ratio: {ratio}",
            f"Tokenizer:         {self.tokenizer}",
        ]
        lines.extend(f"Warning: {w}" for w in self.warnings)
        return "\n".join(lines)


def compare_prompts(
    original: str,
    compressed: str,
    counter: Optional[TokenCounter] = None,
) -> BaselineComparison:
    """Compare original and compressed prompt text with the same tokenizer."""
    if not isinstance(original, str):
        raise TypeError(f"original must be a string, got {type(original).__name__}")
    if not isinstance(compressed, str):
        raise TypeError(f"compressed must be a string, got {type(compressed).__name__}")

    tok = counter or get_token_counter()
    orig_n = tok.count(original)
    comp_n = tok.count(compressed)
    reduction = orig_n - comp_n
    percent = (reduction / orig_n * 100.0) if orig_n > 0 else 0.0
    ratio = (orig_n / comp_n) if comp_n > 0 else None

    warnings: list[str] = []
    if not tok.exact:
        warnings.append(
            f"Token counts are approximate ({tok.name}); install tiktoken and allow it to "
            f"fetch {ENCODING_NAME} once for exact counts."
        )
    if orig_n == 0:
        warnings.append("Original prompt has 0 tokens; reduction percent reported as 0.0.")
    if comp_n == 0 and orig_n > 0:
        warnings.append("Compressed prompt has 0 tokens; compression ratio is undefined.")

    return BaselineComparison(
        original_tokens=orig_n,
        compressed_tokens=comp_n,
        token_reduction=reduction,
        token_reduction_percent=percent,
        compression_ratio=ratio,
        tokenizer=tok.name,
        tokenizer_exact=tok.exact,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------
# Manual check:  python -m src.evaluation.baseline
# --------------------------------------------------------------------------
if __name__ == "__main__":
    tok = get_token_counter()
    print(f"Tokenizer: {tok.name} | exact: {tok.exact}")
    print(f"'Hello world' -> {tok.count('Hello world')} tokens")
    print(f"'' -> {tok.count('')} tokens")
    print()
    print(compare_prompts("Hello world Hello world", "Hello world", tok).summary())

    print("\n[Approximate fallback counter]")
    approx = TokenCounter("approx-regex (not tiktoken)", False, _approx_count)
    print(f"'Hello, world!' -> {approx.count('Hello, world!')} tokens")
    print(compare_prompts("a b c d", "", approx).summary())