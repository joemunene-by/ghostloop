"""Built-in PolicyGate implementations.

Each gate is small and composable. Pipeline order matters: cheap gates first
(deny-list, rate limit) so expensive ones (geofence math, HITL prompt) only
run when the cheap checks pass. All gates here are fail-closed by design.
"""

from .deny_list import DenyListGate
from .force_cap import ForceCapGate
from .geofence import GeofenceGate
from .human_in_the_loop import (
    HumanInTheLoopGate,
    always_approve,
    always_deny,
    cli_approver,
)
from .llm import LLMPolicy, LLMPolicyConfig, LLMPolicyError, llm_policy_loop
from .rate_limit import RateLimitGate
from .vla import (
    ActionDecoder,
    DeltaXYZDecoder,
    VLAPolicy,
    vla_policy_loop,
)
from .workspace import (
    AxisAlignedBox,
    ObstacleAvoidanceGate,
    Sphere,
    WorkspaceModel,
)

__all__ = [
    "DenyListGate",
    "ForceCapGate",
    "GeofenceGate",
    "HumanInTheLoopGate",
    "LLMPolicy",
    "LLMPolicyConfig",
    "LLMPolicyError",
    "RateLimitGate",
    "always_approve",
    "always_deny",
    "cli_approver",
    "llm_policy_loop",
    # VLA
    "ActionDecoder",
    "DeltaXYZDecoder",
    "VLAPolicy",
    "vla_policy_loop",
    # Workspace + obstacles
    "AxisAlignedBox",
    "ObstacleAvoidanceGate",
    "Sphere",
    "WorkspaceModel",
]
