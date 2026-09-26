"""Protection rules for quality-aware prompt compression (Day 2, Step 1).

This module answers one question per node: *how strongly should this unit be
protected from removal, and how should it rank against other units?*

It does NOT decide keep/drop. The compressor (Step 2) combines these
assessments with near-duplicate groups and the retention ratio.

Note on Day 1: the annotated graph carries a ``protection`` score
(1.0 for constraint/requirement/output_format). That value is only a
scoring signal inside the importance formula. The rules here are separate
and also treat ``instruction`` as strongly protected.

Priority definition (documented so it stays explainable)::

    priority = importance + level_bonus[level]

Default bonuses: strong = +1.0, normal = 0.0, low = -0.25. With Day 1
importance in roughly [0, 1], every strong unit outranks every normal unit
by default, and "other" units rank below normal units of similar importance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import networkx as nx


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------
class ProtectionRuleError(Exception):
    """Base class for errors raised by the protection rules."""


class InvalidUnitTypeError(ProtectionRuleError, ValueError):
    """A unit type is not one of the supported Day 1 unit types."""


class InvalidPolicyError(ProtectionRuleError, ValueError):
    """A ProtectionPolicy is inconsistent or incomplete."""


class InvalidGraphError(ProtectionRuleError, ValueError):
    """The graph is not usable (wrong type or missing node attributes)."""


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
class ProtectionLevel(str, Enum):
    STRONG = "strong"
    NORMAL = "normal"
    LOW = "low"


VALID_UNIT_TYPES: frozenset[str] = frozenset(
    {
        "role",
        "instruction",
        "context",
        "constraint",
        "requirement",
        "example",
        "output_format",
        "other",
    }
)

DEFAULT_STRONG_TYPES: frozenset[str] = frozenset(
    {"instruction", "constraint", "requirement", "output_format"}
)
DEFAULT_NORMAL_TYPES: frozenset[str] = frozenset({"role", "context", "example"})
DEFAULT_LOW_TYPES: frozenset[str] = frozenset({"other"})

DEFAULT_LEVEL_BONUS: dict[ProtectionLevel, float] = {
    ProtectionLevel.STRONG: 1.0,
    ProtectionLevel.NORMAL: 0.0,
    ProtectionLevel.LOW: -0.25,
}


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ProtectionPolicy:
    """Configurable mapping from unit type to protection level."""

    strong_types: frozenset[str] = DEFAULT_STRONG_TYPES
    normal_types: frozenset[str] = DEFAULT_NORMAL_TYPES
    low_types: frozenset[str] = DEFAULT_LOW_TYPES
    level_bonus: Mapping[ProtectionLevel, float] = field(
        default_factory=lambda: dict(DEFAULT_LEVEL_BONUS)
    )

    def __post_init__(self) -> None:
        groups = {
            "strong_types": self.strong_types,
            "normal_types": self.normal_types,
            "low_types": self.low_types,
        }
        for name, group in groups.items():
            unknown = set(group) - VALID_UNIT_TYPES
            if unknown:
                raise InvalidPolicyError(
                    f"{name} contains unsupported unit types: {sorted(unknown)}"
                )

        strong, normal, low = self.strong_types, self.normal_types, self.low_types
        overlap = (strong & normal) | (strong & low) | (normal & low)
        if overlap:
            raise InvalidPolicyError(
                f"unit types assigned to more than one level: {sorted(overlap)}"
            )

        missing = VALID_UNIT_TYPES - (strong | normal | low)
        if missing:
            raise InvalidPolicyError(
                f"unit types not assigned to any level: {sorted(missing)}"
            )

        for level in ProtectionLevel:
            if level not in self.level_bonus:
                raise InvalidPolicyError(f"level_bonus is missing {level.value!r}")
            bonus = self.level_bonus[level]
            if isinstance(bonus, bool) or not isinstance(bonus, (int, float)):
                raise InvalidPolicyError(f"level_bonus[{level.value!r}] must be a number")
            if not math.isfinite(bonus):
                raise InvalidPolicyError(f"level_bonus[{level.value!r}] must be finite")

    def level_of(self, unit_type: str) -> ProtectionLevel:
        """Return the protection level for a unit type."""
        if unit_type in self.strong_types:
            return ProtectionLevel.STRONG
        if unit_type in self.normal_types:
            return ProtectionLevel.NORMAL
        if unit_type in self.low_types:
            return ProtectionLevel.LOW
        raise InvalidUnitTypeError(
            f"unsupported unit type {unit_type!r}; expected one of {sorted(VALID_UNIT_TYPES)}"
        )

    def is_protected(self, unit_type: str) -> bool:
        """True only for strongly protected unit types."""
        return self.level_of(unit_type) is ProtectionLevel.STRONG

    def priority(self, unit_type: str, importance: float) -> float:
        """Ranking score = Day 1 importance + level bonus."""
        _validate_importance(importance)
        return float(importance) + float(self.level_bonus[self.level_of(unit_type)])


DEFAULT_POLICY = ProtectionPolicy()


# --------------------------------------------------------------------------
# Assessment result
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ProtectionAssessment:
    """Protection facts about a single node, ready for the compressor and UI."""

    node_id: str
    unit_type: str
    level: ProtectionLevel
    protected: bool
    importance: float
    priority: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "unit_type": self.unit_type,
            "level": self.level.value,
            "protected": self.protected,
            "importance": self.importance,
            "priority": self.priority,
            "reason": self.reason,
        }


_REASONS: dict[ProtectionLevel, str] = {
    ProtectionLevel.STRONG: "{t} is a strongly protected unit type",
    ProtectionLevel.NORMAL: (
        "{t} is a normally retainable unit type; kept or removed based on "
        "importance and redundancy"
    ),
    ProtectionLevel.LOW: "{t} is the lowest-protection unit type; first candidate for removal",
}


# --------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------
def assess_node(
    node_id: str,
    unit_type: str,
    importance: float,
    policy: ProtectionPolicy = DEFAULT_POLICY,
) -> ProtectionAssessment:
    """Assess one node under a policy."""
    if not isinstance(node_id, str) or not node_id:
        raise InvalidGraphError("node_id must be a non-empty string")
    level = policy.level_of(unit_type)  # raises InvalidUnitTypeError
    priority = policy.priority(unit_type, importance)  # validates importance
    return ProtectionAssessment(
        node_id=node_id,
        unit_type=unit_type,
        level=level,
        protected=level is ProtectionLevel.STRONG,
        importance=float(importance),
        priority=priority,
        reason=_REASONS[level].format(t=unit_type),
    )


def assess_graph(
    graph: nx.Graph,
    policy: ProtectionPolicy = DEFAULT_POLICY,
) -> dict[str, ProtectionAssessment]:
    """Assess every node of an annotated Day 1 graph.

    Reads the ``type`` and ``importance`` node attributes. Returns a dict
    keyed by node id, in graph node order. An empty graph returns ``{}``;
    the compressor decides how to treat that case.
    """
    if not isinstance(graph, nx.Graph):
        raise InvalidGraphError(f"expected a networkx.Graph, got {type(graph).__name__}")

    result: dict[str, ProtectionAssessment] = {}
    for key, data in graph.nodes(data=True):
        for attr in ("type", "importance"):
            if attr not in data:
                raise InvalidGraphError(
                    f"node {key!r} is missing required attribute {attr!r}; "
                    "run annotate_graph() first"
                )
        node_id = data.get("node_id", key)
        result[node_id] = assess_node(node_id, data["type"], data["importance"], policy)
    return result


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _validate_importance(importance: float) -> None:
    if isinstance(importance, bool) or not isinstance(importance, (int, float)):
        raise ProtectionRuleError(
            f"importance must be a number, got {type(importance).__name__}"
        )
    if not math.isfinite(importance) or importance < 0:
        raise ProtectionRuleError(f"importance must be finite and >= 0, got {importance}")