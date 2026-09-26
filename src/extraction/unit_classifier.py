"""Rule-based information-unit classification.

Input : InformationUnit(s) from the segmenter
Output: ClassifiedUnit(s) with type, confidence, and the rules that fired

Method (deterministic, explainable, no LLM):
  1. Each rule = (name, target type, weight, regex).
  2. Every matching rule adds its weight to its target type.
  3. Highest total wins; ties are broken by a fixed priority order.
  4. No rule matched -> 'other' (confidence 0.0).

Confidence = winning score / total matched score. It measures how much of the
matched evidence agrees with the winner; it is NOT a probability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

from src.preprocessing.segmenter import InformationUnit, SegmentationResult

UnitType = Literal[
    "role", "instruction", "context", "constraint",
    "requirement", "example", "output_format", "other",
]
UNIT_TYPES: tuple[UnitType, ...] = (
    "role", "instruction", "context", "constraint",
    "requirement", "example", "output_format", "other",
)

# Tie-breaking: more specific types first. 'other' never receives evidence.
_PRIORITY: tuple[str, ...] = (
    "constraint", "requirement", "output_format", "example",
    "role", "instruction", "context",
)

HEADING_WEIGHT = 3.0


class ClassificationError(ValueError):
    """Raised when there is nothing valid to classify."""


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassifiedUnit:
    """An information unit together with its predicted type and evidence."""

    unit: InformationUnit
    unit_type: UnitType
    confidence: float
    matched_rules: tuple[str, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def unit_id(self) -> str:
        return self.unit.unit_id

    @property
    def text(self) -> str:
        return self.unit.text

    def explain(self) -> str:
        """Human-readable justification for the classification."""
        if not self.matched_rules:
            return f"{self.unit_id}: other (no rule matched)"
        ranked = sorted(self.scores.items(), key=lambda kv: -kv[1])
        breakdown = ", ".join(f"{k}={v:g}" for k, v in ranked)
        return (
            f"{self.unit_id}: {self.unit_type} (confidence {self.confidence:.2f}) "
            f"via {', '.join(self.matched_rules)} | scores: {breakdown}"
        )


# --------------------------------------------------------------------------
# Rule tables
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _Rule:
    name: str
    label: str
    weight: float
    pattern: re.Pattern[str]


def _r(name: str, label: str, weight: float, pattern: str) -> _Rule:
    return _Rule(name, label, weight, re.compile(pattern, re.IGNORECASE))


_IMPERATIVE_VERBS = (
    "write|create|summari[sz]e|explain|list|generate|analy[sz]e|translate|describe|"
    "provide|give|compare|identify|extract|classify|review|draft|answer|help|plan|"
    "find|suggest|recommend|produce|develop|design|build|calculate|evaluate|rewrite|"
    "edit|proofread|convert|tell|show|take|read|check|determine|outline|brainstorm|"
    "solve|predict|rank|select|choose|complete|fix|debug|refactor|implement|test|"
    "respond|reply|discuss|focus|consider|assume|imagine|think|start|begin"
)

_TEXT_RULES: tuple[_Rule, ...] = (
    # ---- role ----
    _r("you_are_role", "role", 3, r"^you are (?:a|an|the|my|our)\b"),
    _r("act_as", "role", 3, r"^(?:please\s+)?act as\b|\byou will (?:act|play|serve|work) as\b"),
    _r("role_played", "role", 3, r"\b(?:acting|playing) the role of\b|^as an? .{1,60}?, you\b"),
    _r("your_role_is", "role", 3, r"^your (?:role|job|persona) is\b"),
    _r("role_prefix", "role", 3, r"^(?:role|persona)\s*:"),
    # ---- instruction ----
    _r("task_prefix", "instruction", 3, r"^(?:task|instructions?|goal|objective|question)\s*:"),
    _r("your_task", "instruction", 2, r"^your (?:task|goal|objective|mission) is\b"),
    _r("imperative_start", "instruction", 2,
       rf"^(?:(?:please|also|then|next|finally|first|and|now)[,\s]+)*(?:{_IMPERATIVE_VERBS})\b"),
    _r("question", "instruction", 2, r"\?\s*$"),
    _r("request_phrase", "instruction", 2,
       r"\b(?:(?:can|could|would) you|i (?:need|want|would like) you to)\b"),
    # ---- requirement ----
    _r("modal_must", "requirement", 2,
       r"\b(?:must|shall|need to|needs to|has to|have to|required to|is required|are required)\b(?!\s+not\b)"),
    _r("modal_should", "requirement", 1, r"\bshould\b(?!\s+not\b)"),
    _r("ensure_start", "requirement", 2, r"^(?:include|ensure|make sure|always|remember to|be sure to)\b"),
    _r("ensure_inline", "requirement", 1, r"\b(?:make sure|ensure that|at least)\b"),
    # ---- constraint ----
    _r("prohibition", "constraint", 2,
       r"\b(?:must not|mustn't|should not|shouldn't|do not|don't|cannot|can't|never)\b"),
    _r("avoid_start", "constraint", 2, r"^(?:avoid|refrain from|without|no|only)\b"),
    _r("limit_bound", "constraint", 2,
       r"\b(?:at most|no more than|up to|maximum|max|not exceed|exceed|within|fewer than|"
       r"less than|limited to|limit)\b|\bunder \$?\d"),
    _r("only_restriction", "constraint", 1, r"\bonly\b"),
    # ---- output_format ----
    _r("format_prefix", "output_format", 3,
       r"^(?:output(?: format)?|response format|format|output structure)\s*:"),
    _r("respond_in_format", "output_format", 2,
       r"\b(?:respond|reply|answer|output|return|format|structure)\b.{0,40}\b(?:in|as|with|using)\b"
       r".{0,40}\b(?:json|markdown|csv|yaml|xml|html|table|bullet(?:ed)? points?|bullets|list|"
       r"paragraphs?|sentences?|format|code block)\b"),
    _r("format_noun", "output_format", 2,
       r"\b(?:output|response|answer|reply) (?:format|should be|must be|structure)\b|"
       r"\bformat (?:the|your|it)\b|\bformatted as\b"),
    _r("format_keyword", "output_format", 1,
       r"\b(?:json|markdown|csv|yaml|xml|html|tsv|bullet points?|numbered list|table)\b"),
    # ---- example ----
    _r("example_prefix", "example", 3,
       r"^(?:(?:for example|for instance|examples?|sample)\b|e\.g\.)"),
    # ---- context ----
    _r("context_prefix", "context", 3,
       r"^(?:context|background|scenario|situation|audience|note)\s*:"),
    _r("context_reference", "context", 2,
       r"\b(?:the following (?:text|document|article|passage|data|email|code|conversation|report)|"
       r"given (?:the|a|an)|here is|here's|below is|attached)\b"),
    _r("background_statement", "context", 1, r"^(?:the|our|this|these|those|we|i|my|there)\b"),
)

# Checked in this order; the FIRST matching heading label is used.
_HEADING_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pat, re.IGNORECASE))
    for label, pat in (
        ("example", r"\bexamples?\b|\bsamples?\b|\bdemonstrations?\b"),
        ("constraint", r"\bconstraints?\b|\brestrictions?\b|\blimitations?\b|\brules\b|"
                       r"\bdo not\b|\bdon'?ts?\b|\bavoid\b|\bboundaries\b"),
        ("requirement", r"\brequirements?\b|\bmust[- ]haves?\b|\bcriteria\b|"
                        r"\bguidelines\b|\bspecifications?\b"),
        ("output_format", r"\boutputs?\b|\bformat\b|\bresponse\b|\bdeliverables?\b"),
        ("role", r"\brole\b|\bpersona\b|\bidentity\b"),
        ("context", r"\bcontext\b|\bbackground\b|\bscenario\b|\bsituation\b|"
                    r"\baudience\b|\babout\b"),
        ("instruction", r"\btasks?\b|\binstructions?\b|\bsteps?\b|\bgoals?\b|\bobjectives?\b"),
    )
)

_FENCE_LANG = re.compile(r"^\s*(?:```|~~~)\s*([A-Za-z0-9_+-]+)?")
_FORMAT_LANGS = {"json", "yaml", "yml", "xml", "csv", "markdown", "md", "html", "tsv"}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def classify_unit(unit: InformationUnit) -> ClassifiedUnit:
    """Classify a single information unit.

    Raises:
        TypeError: if ``unit`` is not an InformationUnit.
    """
    if not isinstance(unit, InformationUnit):
        raise TypeError(f"Expected InformationUnit, got {type(unit).__name__}.")

    scores: dict[str, float] = {}
    matched: list[str] = []

    def add(name: str, label: str, weight: float) -> None:
        scores[label] = scores.get(label, 0.0) + weight
        matched.append(name)

    # 1. heading evidence (from the segmenter)
    if unit.parent_heading:
        for label, pattern in _HEADING_RULES:
            if pattern.search(unit.parent_heading):
                add(f"heading:{label}", label, HEADING_WEIGHT)
                break

    # 2. content evidence
    if unit.kind == "code_block":
        m = _FENCE_LANG.match(unit.text)
        lang = (m.group(1) or "").lower() if m else ""
        if lang in _FORMAT_LANGS:
            add(f"fence_language:{lang}", "output_format", 1.0)
        else:
            add("code_block_default", "context", 1.0)
    else:
        for rule in _TEXT_RULES:
            if rule.pattern.search(unit.text):
                add(rule.name, rule.label, rule.weight)

    if not scores:
        return ClassifiedUnit(unit=unit, unit_type="other", confidence=0.0)

    total = sum(scores.values())
    best = max(scores, key=lambda lab: (scores[lab], -_PRIORITY.index(lab)))
    return ClassifiedUnit(
        unit=unit,
        unit_type=best,  # type: ignore[arg-type]
        confidence=round(scores[best] / total, 2),
        matched_rules=tuple(matched),
        scores=dict(scores),
    )


def classify_units(
    units: SegmentationResult | Iterable[InformationUnit],
) -> tuple[ClassifiedUnit, ...]:
    """Classify many units, preserving their order.

    Raises:
        ClassificationError: if no units are provided.
        TypeError: if an element is not an InformationUnit.
    """
    items = units.units if isinstance(units, SegmentationResult) else tuple(units)
    if not items:
        raise ClassificationError("No information units to classify.")
    return tuple(classify_unit(u) for u in items)


def type_counts(classified: Iterable[ClassifiedUnit]) -> dict[str, int]:
    """Count units per type (non-zero types only, in UNIT_TYPES order)."""
    counts = {t: 0 for t in UNIT_TYPES}
    for c in classified:
        counts[c.unit_type] += 1
    return {t: n for t, n in counts.items() if n}


# --------------------------------------------------------------------------
# Manual check:  python -m src.extraction.unit_classifier
# --------------------------------------------------------------------------
if __name__ == "__main__":
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
    print(f"Units: {len(classified)}")
    for c in classified:
        print(f"{c.unit_id} | {c.unit_type:<13} | conf {c.confidence:.2f} | "
              f"{', '.join(c.matched_rules) or '-'}")
    print(f"Counts: {type_counts(classified)}")