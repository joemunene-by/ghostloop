"""Deny-list gate: hard-block named primitives without an explicit override.

Useful for staging environments (e.g. block ``open_gripper`` until vision is
calibrated) and for incident response (block all ``move_*`` primitives in one
config flip while you investigate)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core import Decision, Intent, Primitive


@dataclass
class DenyListGate:
    """Reject any Intent whose primitive name is in ``denied``.

    Set membership lookup, fast enough to put first in the pipeline.
    """

    denied: set[str] = field(default_factory=set)
    name: str = "deny_list"

    def check(self, intent: Intent, primitive: Primitive) -> Decision:
        if primitive.name in self.denied:
            return Decision.deny(
                self.name,
                f"{primitive.name} is on the deny list",
            )
        return Decision.allow(self.name, "not on deny list")
