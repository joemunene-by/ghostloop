"""Cross-embodiment primitive library — same skill name, different robot.

In ghostloop a Primitive is a backend-bound callable. That works
great for one robot, but a real fleet has many: Franka Panda + UR5e
arms in the lab, Spot quadrupeds in the field, Stretch RE3 mobile
manipulator on the factory floor. Each implements ``pick`` differently
(7-DOF arm IK vs mobile-base + 6-DOF arm vs gripper-on-base) yet the
HIGH-LEVEL skill is the same.

``MorphologyRegistry`` is keyed by ``(morphology, primitive_name)``
and produces the appropriate Primitive at build time:

    reg = MorphologyRegistry()
    reg.register("franka", "pick", franka_pick_factory)
    reg.register("ur5e",   "pick", ur5e_pick_factory)
    reg.register("spot",   "pick", spot_pick_factory)

    primitives = reg.build("franka", ["move_to", "pick", "place"])

Same downstream code, different robots. Pairs with the SkillGraph from
v0.10 — the SAME skill graph (with prereqs and refines edges) drives
every morphology.

Pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..core import Primitive


PrimitiveFactory = Callable[[], Primitive]


class MorphologyError(KeyError):
    """Raised when a morphology / primitive lookup fails."""


@dataclass
class MorphologyRegistry:
    """Two-level registry keyed by (morphology, primitive name)."""

    _factories: dict[tuple[str, str], PrimitiveFactory] = field(
        default_factory=dict,
    )

    def register(
        self,
        morphology: str,
        primitive_name: str,
        factory: PrimitiveFactory,
    ) -> None:
        self._factories[(morphology, primitive_name)] = factory

    def register_many(
        self,
        morphology: str,
        factories: dict[str, PrimitiveFactory],
    ) -> None:
        for name, fac in factories.items():
            self.register(morphology, name, fac)

    def build_one(self, morphology: str, primitive_name: str) -> Primitive:
        key = (morphology, primitive_name)
        if key not in self._factories:
            raise MorphologyError(
                f"no primitive {primitive_name!r} registered for morphology "
                f"{morphology!r}; available: "
                f"{sorted(p for m, p in self._factories if m == morphology)}"
            )
        return self._factories[key]()

    def build(self, morphology: str, primitive_names: list[str]) -> list[Primitive]:
        return [self.build_one(morphology, name) for name in primitive_names]

    def supported(self, morphology: str) -> list[str]:
        return sorted(p for m, p in self._factories.keys() if m == morphology)

    def morphologies(self) -> list[str]:
        return sorted({m for m, _ in self._factories.keys()})

    def coverage(self, morphology: str, required: list[str]) -> dict:
        """Report which required primitives ARE / AREN'T supported by ``morphology``."""
        supported = set(self.supported(morphology))
        return {
            "morphology": morphology,
            "required": list(required),
            "covered": [n for n in required if n in supported],
            "missing": [n for n in required if n not in supported],
        }
