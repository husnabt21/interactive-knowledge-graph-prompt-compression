"""Prompt segmentation: split a cleaned prompt into information units.

Input : CleanedPrompt (or a plain string)
Output: SegmentationResult (tuple of InformationUnit + warnings)

Structure-aware rules (deterministic, no LLM):
  * fenced code blocks        -> one 'code_block' unit
  * bullet / numbered items   -> one 'list_item' unit each
  * short "Label:" lines and
    Markdown '#' headings     -> not units; stored as parent_heading
  * remaining prose           -> sentence-level units
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from src.preprocessing.cleaner import CleanedPrompt, EmptyPromptError

UnitKind = Literal["sentence", "list_item", "code_block"]


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------
class SegmentationError(ValueError):
    """Raised when a prompt yields no information units."""


# --------------------------------------------------------------------------
# Configuration and data structures
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SegmenterConfig:
    min_words_per_fragment: int = 2   # shorter sentence fragments get merged
    max_label_words: int = 3          # "Output format:" style labels


@dataclass(frozen=True)
class InformationUnit:
    """One atomic piece of prompt information (becomes a graph node)."""

    unit_id: str                      # "N01", "N02", ...
    index: int                        # 0-based order in the prompt
    text: str
    kind: UnitKind
    line_start: int                   # 1-based line in the cleaned text
    parent_heading: str | None = None


@dataclass(frozen=True)
class SegmentationResult:
    units: tuple[InformationUnit, ...]
    warnings: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Regexes / constants
# --------------------------------------------------------------------------
_FENCE_MARKERS = ("```", "~~~")
_LIST_ITEM = re.compile(r"^(\s*)(?:[-*+\u2022]|\d{1,3}[.)])\s+(\S.*)$")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(\S.*?)\s*#*\s*$")
_SENTENCE_BOUNDARY = re.compile(
    r"""(?:(?<=[.!?])|(?<=[.!?]["')\]]))\s+(?=["'(\[]?[A-Z0-9])"""
)
_ABBREVIATIONS = {
    "e.g.", "i.e.", "vs.", "mr.", "mrs.", "ms.", "dr.", "prof.",
    "fig.", "approx.", "cf.",
}


# --------------------------------------------------------------------------
# Sentence-level helpers
# --------------------------------------------------------------------------
def _is_soft_wrap(prev: str, nxt: str) -> bool:
    """True if `nxt` continues `prev` (line wrapped mid-sentence)."""
    return not prev.endswith((".", "!", "?", ":", ";")) and nxt[:1].islower()


def _logical_lines(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    merged: list[tuple[int, str]] = []
    for line_no, text in lines:
        if merged and _is_soft_wrap(merged[-1][1], text):
            merged[-1] = (merged[-1][0], f"{merged[-1][1]} {text}")
        else:
            merged.append((line_no, text))
    return merged


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, re-joining after known abbreviations."""
    parts = [p.strip() for p in _SENTENCE_BOUNDARY.split(text) if p.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and merged[-1].split()[-1].lower() in _ABBREVIATIONS:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _merge_short(fragments: list[str], min_words: int) -> list[str]:
    out: list[str] = []
    for frag in fragments:
        if out and len(frag.split()) < min_words:
            out[-1] = f"{out[-1]} {frag}"
        else:
            out.append(frag)
    if len(out) > 1 and len(out[0].split()) < min_words:
        out[1] = f"{out[0]} {out[1]}"
        out.pop(0)
    return out


def _split_paragraph(
    lines: list[tuple[int, str]], cfg: SegmenterConfig
) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for line_no, text in _logical_lines(lines):
        frags = _merge_short(_split_sentences(text), cfg.min_words_per_fragment)
        result.extend((line_no, f) for f in frags)
    return result


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def segment_prompt(
    cleaned: CleanedPrompt | str, config: SegmenterConfig | None = None
) -> SegmentationResult:
    """Split a cleaned prompt into information units.

    Raises:
        TypeError: invalid input type.
        EmptyPromptError: empty text.
        SegmentationError: no units could be extracted.
    """
    cfg = config or SegmenterConfig()

    if isinstance(cleaned, CleanedPrompt):
        text = cleaned.text
    elif isinstance(cleaned, str):
        text = cleaned
    else:
        raise TypeError(
            f"Expected CleanedPrompt or str, got {type(cleaned).__name__}."
        )
    if not text.strip():
        raise EmptyPromptError("Cannot segment an empty prompt.")

    warnings: list[str] = []
    raw: list[tuple[UnitKind, str, int, str | None]] = []

    heading: str | None = None
    heading_scope: str | None = None            # "md" or "label"
    label_pending: tuple[str, int] | None = None  # label with no content yet
    para: list[tuple[int, str]] = []
    item: list | None = None                    # [line_no, text, heading]

    def emit(kind: UnitKind, txt: str, line_no: int, head: str | None) -> None:
        nonlocal label_pending
        raw.append((kind, txt, line_no, head))
        label_pending = None

    def flush_item() -> None:
        nonlocal item
        if item is not None:
            emit("list_item", item[1], item[0], item[2])
            item = None

    def flush_para() -> None:
        nonlocal para
        for line_no, seg in _split_paragraph(para, cfg):
            emit("sentence", seg, line_no, heading)
        para = []

    def flush_all() -> None:
        flush_item()
        flush_para()

    def set_heading(label: str, line_no: int, scope: str) -> None:
        nonlocal heading, heading_scope, label_pending
        flush_all()
        if label_pending is not None:           # previous label had no content
            emit("sentence", label_pending[0], label_pending[1], None)
        heading, heading_scope = label, scope
        label_pending = (label, line_no)

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        line_no = i + 1
        stripped = line.strip()

        # ---- fenced code block ----
        if stripped.startswith(_FENCE_MARKERS):
            flush_all()
            marker = stripped[:3]
            block = [line.rstrip()]
            i += 1
            closed = False
            while i < len(lines):
                block.append(lines[i].rstrip())
                if lines[i].strip().startswith(marker):
                    closed = True
                    break
                i += 1
            if not closed:
                warnings.append(f"Unclosed code fence starting at line {line_no}.")
            emit("code_block", "\n".join(block), line_no, heading)
            i += 1
            continue

        # ---- blank line ----
        if not stripped:
            flush_all()
            if heading_scope == "label" and label_pending is None:
                heading, heading_scope = None, None
            i += 1
            continue

        # ---- Markdown heading ----
        md = _MD_HEADING.match(line)
        if md:
            set_heading(md.group(1).strip(), line_no, "md")
            i += 1
            continue

        # ---- list item ----
        li = _LIST_ITEM.match(line)
        if li:
            flush_para()
            flush_item()
            item = [line_no, li.group(2).strip(), heading]
            i += 1
            continue

        # ---- short "Label:" line ----
        if stripped.endswith(":") and len(stripped.split()) <= cfg.max_label_words:
            set_heading(stripped[:-1].strip(), line_no, "label")
            i += 1
            continue

        # ---- continuation of a list item (indented text) ----
        if item is not None and line.startswith(" "):
            item[1] = f"{item[1]} {stripped}"
            i += 1
            continue

        # ---- normal paragraph line ----
        flush_item()
        para.append((line_no, stripped))
        i += 1

    flush_all()
    if label_pending is not None:               # trailing label with no content
        emit("sentence", label_pending[0], label_pending[1], None)

    if not raw:
        raise SegmentationError("No information units could be extracted.")

    width = max(2, len(str(len(raw))))
    units = tuple(
        InformationUnit(
            unit_id=f"N{idx + 1:0{width}d}",
            index=idx,
            text=txt,
            kind=kind,
            line_start=line_no,
            parent_heading=head,
        )
        for idx, (kind, txt, line_no, head) in enumerate(raw)
    )

    if len(units) == 1:
        warnings.append("Only one information unit found; the graph will have no edges.")

    seen: dict[str, str] = {}
    for u in units:
        key = " ".join(u.text.lower().split())
        if key in seen:
            warnings.append(f"{u.unit_id} duplicates {seen[key]} (kept, not removed).")
        else:
            seen[key] = u.unit_id

    return SegmentationResult(units=units, warnings=tuple(warnings))


# --------------------------------------------------------------------------
# Manual check:  python -m src.preprocessing.segmenter
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from src.preprocessing.cleaner import clean_prompt

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
    result = segment_prompt(clean_prompt(sample))
    print(f"Units: {len(result.units)}")
    for u in result.units:
        print(f"{u.unit_id} | {u.kind:<10} | L{u.line_start:<2} | {u.parent_heading} | {u.text!r}")
    print(f"Warnings: {list(result.warnings)}")