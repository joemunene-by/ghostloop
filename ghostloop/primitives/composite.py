"""Composite primitives — sequence existing primitives behind a single name.

Real robots have macros: "approach_grasp" = (move_to_pre_grasp,
descend_to_grasp_pose, close_gripper, lift). Defining each as one
``Primitive`` keeps the registry small and lets policies reason at the
right level of abstraction.

A composite primitive is itself a Primitive; its ``call`` dispatches the
underlying primitives sequentially through whatever Backend it received.
Each sub-primitive's safety pipeline checks happen at the OUTER layer
(when the composite intent is dispatched), since the composite just
calls the sub-primitives directly. For per-substep gating, decompose
into separate Intents in a TaskPlanner instead.
"""

from __future__ import annotations

from typing import Callable

from ..core import Backend, Primitive, Result, ResultStatus


def composite_primitive(
    name: str,
    steps: list[Primitive],
    *,
    description: str = "",
    arg_schema: dict[str, str] | None = None,
    map_args: Callable[[dict, int, Primitive], dict] | None = None,
) -> Primitive:
    """Build a composite primitive from a sequence of underlying primitives.

    Args:
        name: name for the composite.
        steps: ordered list of Primitives to call in sequence.
        description: human-readable docstring for the composite.
        arg_schema: schema the composite accepts. Default empty.
        map_args: optional callable ``(composite_args, step_idx, step_primitive)
                  -> step_args``. Default: pass the composite_args
                  through unchanged.

    Stops at the first non-OK result and returns that result with a
    ``failed_at`` field in the observation. Each substep's observation
    is preserved under ``substep_<i>_<name>`` keys.
    """
    def _call(backend: Backend, **kwargs) -> Result:
        observation: dict = {"composite": name, "n_steps": len(steps)}
        for i, step in enumerate(steps):
            mapped = map_args(kwargs, i, step) if map_args else dict(kwargs)
            try:
                step_result = step.call(backend, **mapped)
            except Exception as exc:  # noqa: BLE001
                return Result(
                    status=ResultStatus.ERROR,
                    observation={**observation, "failed_at": step.name, "step_index": i},
                    message=f"{name}.{step.name}: {type(exc).__name__}: {exc}",
                )
            observation[f"substep_{i}_{step.name}"] = step_result.observation
            if not step_result.ok:
                return Result(
                    status=step_result.status,
                    observation={
                        **observation,
                        "failed_at": step.name,
                        "step_index": i,
                    },
                    message=f"{name}.{step.name}: {step_result.message}",
                )
        return Result(
            status=ResultStatus.OK,
            observation=observation,
            message=f"{name}: {len(steps)} substeps complete",
        )

    return Primitive(
        name=name,
        call=_call,
        description=description or f"Composite of {len(steps)} primitives.",
        arg_schema=arg_schema or {},
    )
