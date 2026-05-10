"""Declarative safety properties.

Runtime gates inspect ONE intent at a time. Properties are higher-level
invariants that span the trace — they let you assert "the robot never
holds two objects at once", "the end-effector stays inside the workspace
across the entire episode", "no two consecutive steps emit the same
intent without an intervening observation". Properties don't block at
the gate layer; they audit traces and surface violations.

Use cases:
  - CI gates ("ship-blocking" properties evaluated against every recorded
    trace before a deploy).
  - Bench scoring beyond pass/fail ("how many violations per episode").
  - Post-incident analysis ("which property broke first when the
    pick-and-place run went off the rails?").
"""

from .core import (
    Property,
    PropertyResult,
    PropertyEngine,
    Severity,
)
from .builtins import (
    NeverHoldsTwoObjects,
    NeverExceedsRate,
    NoConsecutiveDuplicateIntents,
    StaysInsideWorkspace,
)
from .combinators import AndProperty, NotProperty, OrProperty
from .mining import MinedProperty, mine_properties
from .temporal import (
    Always,
    Eventually,
    EventPredicate,
    Until,
    decision_action,
    intent_named,
    result_status,
    state_field_below,
)

__all__ = [
    "Property",
    "PropertyResult",
    "PropertyEngine",
    "Severity",
    "NeverHoldsTwoObjects",
    "NeverExceedsRate",
    "NoConsecutiveDuplicateIntents",
    "StaysInsideWorkspace",
    "AndProperty",
    "OrProperty",
    "NotProperty",
    "Always",
    "Eventually",
    "Until",
    "EventPredicate",
    "intent_named",
    "decision_action",
    "result_status",
    "state_field_below",
    "MinedProperty",
    "mine_properties",
]
