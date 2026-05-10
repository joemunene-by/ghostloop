"""Mock-backend manipulation primitives: pick and place."""

from __future__ import annotations

from ..core import MockBackend, Primitive, Result, ResultStatus


def _pick_call(backend: MockBackend, object_id: str) -> Result:
    if backend.held_object is not None:
        return Result(
            status=ResultStatus.ERROR,
            message=f"already holding {backend.held_object!r}; place it first",
        )
    backend.held_object = str(object_id)
    return Result(
        status=ResultStatus.OK,
        observation={
            "picked": str(object_id),
            "at_pose": list(backend.position),
        },
        message=f"picked {object_id!r}",
    )


def pick() -> Primitive:
    """Grasp an object by id at the current pose.

    Mock just records the held object. Real backends would close a gripper
    with force-feedback, verify grasp via tactile or vision, and surface
    failure modes (slipped, dropped, wrong object).
    """
    return Primitive(
        name="pick",
        call=_pick_call,
        description="Grasp an object by id at the current end-effector pose.",
        arg_schema={"object_id": "str"},
    )


def _place_call(backend: MockBackend) -> Result:
    if backend.held_object is None:
        return Result(
            status=ResultStatus.ERROR,
            message="nothing to place; pick something first",
        )
    placed = backend.held_object
    backend.held_object = None
    return Result(
        status=ResultStatus.OK,
        observation={
            "placed": placed,
            "at_pose": list(backend.position),
        },
        message=f"placed {placed!r}",
    )


def place() -> Primitive:
    """Release the currently held object at the current pose."""
    return Primitive(
        name="place",
        call=_place_call,
        description="Release the held object at the current end-effector pose.",
        arg_schema={},
    )
