"""HumanInTheLoopGate: block selected intents until an external reviewer approves.

Two concrete approval modes ship in v0.3:

  AlwaysAllow / AlwaysDeny — for tests + dry runs.
  CallableApprover — wrap any synchronous predicate ``(intent, primitive) -> bool``.
                      Common patterns: a CLI prompt, a Slack webhook, a
                      Next.js dashboard polling a queue.

The gate only applies HITL to a configurable set of primitive names — so
``observe`` flows freely while ``open_gripper`` requires sign-off."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..core import Decision, Intent, Primitive


@dataclass
class HumanInTheLoopGate:
    """Block intents whose primitive name is in ``requires_approval``
    until ``approver(intent, primitive)`` returns True.

    Args:
        requires_approval: set of primitive names that need HITL.
        approver: callable that returns True/False. Synchronous; the
            runtime blocks on it. Wrap async approvals in a thread or
            poll a queue with a sync reader.
        name: gate name shown in the trace.
    """

    requires_approval: set[str] = field(default_factory=set)
    approver: Callable[[Intent, Primitive], bool] = field(
        default=lambda intent, primitive: False
    )
    name: str = "hitl"

    def check(self, intent: Intent, primitive: Primitive) -> Decision:
        if primitive.name not in self.requires_approval:
            return Decision.allow(self.name, "no HITL required for this primitive")
        try:
            ok = bool(self.approver(intent, primitive))
        except Exception as exc:  # noqa: BLE001
            return Decision.deny(
                self.name,
                f"approver raised {type(exc).__name__}: {exc}",
            )
        if not ok:
            return Decision.deny(
                self.name,
                f"human reviewer denied {primitive.name!r} call",
            )
        return Decision.allow(self.name, f"human reviewer approved {primitive.name!r}")


def always_approve(intent: Intent, primitive: Primitive) -> bool:
    """Approver that always allows. Useful in tests and dry runs."""
    return True


def always_deny(intent: Intent, primitive: Primitive) -> bool:
    """Approver that always denies. Useful for incident response (block all
    HITL-required primitives without removing the gate)."""
    return False


def cli_approver(intent: Intent, primitive: Primitive) -> bool:
    """Synchronous CLI prompt approver — blocks the runtime on stdin.

    For interactive development. Production deployments will plug a
    Slack-webhook or dashboard-poller into ``HumanInTheLoopGate.approver``.
    """
    print(
        f"[hitl] {primitive.name}({intent.args}) — rationale: {intent.rationale!r}",
        flush=True,
    )
    while True:
        ans = input("approve? [y/N] ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("", "n", "no"):
            return False
        print("please type y or n")
