"""GymnasiumBackend: any Farama Gymnasium / Gym-style environment as a Backend.

Massive ecosystem unlock. Gymnasium ships hundreds of robotics + RL envs
under one ``env.step(action) -> (obs, reward, terminated, truncated, info)``
contract: classic control, MuJoCo locomotion, robotics manipulation,
Atari, pybullet-gym ports, custom envs from research papers. Wrapping
that surface as a ghostloop Backend means every agent loop / safety
pipeline / bench harness in this repo immediately works against the
entire ecosystem.

Conditional import — package itself doesn't depend on gymnasium.
``GymnasiumBackend(env_id="HalfCheetah-v5")`` raises ImportError with
install hint at construction time only.

  pip install ghostloop[gym]    # Farama Gymnasium
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core import Primitive, Result, ResultStatus


def gymnasium_available() -> bool:
    """True iff ``import gymnasium`` succeeds."""
    try:
        import gymnasium  # noqa: F401
        return True
    except ImportError:
        try:
            import gym  # noqa: F401  (older 'gym' package)
            return True
        except ImportError:
            return False


_GYM_INSTALL_HINT = (
    "GymnasiumBackend requires gymnasium (or legacy gym).\n"
    "  pip install gymnasium\n"
    "or  pip install ghostloop[gym]\n"
    "Env catalog: https://gymnasium.farama.org/environments/"
)


@dataclass
class GymnasiumBackend:
    """Wrap a Gymnasium env as a ghostloop Backend.

    Args:
        env_id: env registry id (e.g. ``"HalfCheetah-v5"``, ``"FetchReach-v3"``).
        render_mode: optional render mode (``"human"`` / ``"rgb_array"``).
        seed: deterministic seeding for reproducibility.
        kwargs: extra kwargs forwarded to ``gym.make(env_id, **kwargs)``.
        name: friendly name for traces.
    """

    env_id: str
    render_mode: str | None = None
    seed: int | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = "gymnasium"

    _gym: Any = field(default=None, init=False, repr=False)
    _env: Any = field(default=None, init=False, repr=False)
    _last_obs: Any = field(default=None, init=False, repr=False)
    _last_info: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _terminated: bool = field(default=False, init=False, repr=False)
    _truncated: bool = field(default=False, init=False, repr=False)
    _episode_reward: float = field(default=0.0, init=False, repr=False)
    _step_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import gymnasium as gym  # type: ignore
        except ImportError:
            try:
                import gym  # type: ignore
            except ImportError as e:
                raise ImportError(_GYM_INSTALL_HINT) from e
        self._gym = gym
        make_kwargs = dict(self.kwargs)
        if self.render_mode:
            make_kwargs.setdefault("render_mode", self.render_mode)
        self._env = gym.make(self.env_id, **make_kwargs)
        if self.seed is not None:
            self._last_obs, self._last_info = self._env.reset(seed=self.seed)
        else:
            self._last_obs, self._last_info = self._env.reset()

    def __del__(self) -> None:
        try:
            if self._env is not None:
                self._env.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Backend Protocol surface.
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        # Best-effort serialise the observation: tuples / numpy arrays / dicts /
        # primitives. Falls back to repr() for exotic shapes.
        return {
            "backend": self.name,
            "env_id": self.env_id,
            "step": self._step_count,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "episode_reward": float(self._episode_reward),
            "observation": _serialise(self._last_obs),
            "info_keys": sorted(self._last_info.keys()) if isinstance(self._last_info, dict) else [],
        }

    # ------------------------------------------------------------------
    # Primitive call helpers.
    # ------------------------------------------------------------------

    def apply_action(self, action: Any) -> dict[str, Any]:
        """Step the env once with the supplied action. Returns the observation
        + reward + terminated + truncated + info dict — primitives wrap this."""
        if self._terminated or self._truncated:
            return {
                "reward": 0.0,
                "terminated": self._terminated,
                "truncated": self._truncated,
                "info": _serialise(self._last_info),
                "observation": _serialise(self._last_obs),
                "noop": True,
            }
        out = self._env.step(action)
        # Gymnasium 5-tuple: (obs, reward, terminated, truncated, info).
        # Legacy gym: 4-tuple (obs, reward, done, info).
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated, truncated = bool(done), False
        self._last_obs = obs
        self._last_info = info if isinstance(info, dict) else {}
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        self._episode_reward += float(reward)
        self._step_count += 1
        return {
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": _serialise(self._last_info),
            "observation": _serialise(obs),
        }

    def reset_env(self) -> dict[str, Any]:
        kwargs = {"seed": self.seed} if self.seed is not None else {}
        self._last_obs, self._last_info = self._env.reset(**kwargs)
        self._terminated = False
        self._truncated = False
        self._episode_reward = 0.0
        self._step_count = 0
        return {"observation": _serialise(self._last_obs), "info": _serialise(self._last_info)}


def _serialise(obj: Any) -> Any:
    """Best-effort JSON-ish serialisation for Gym observations / info dicts."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialise(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _serialise(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, "__iter__"):
        try:
            return [_serialise(x) for x in obj]
        except Exception:
            pass
    return repr(obj)


# ----------------------------------------------------------------------
# Primitives bound to GymnasiumBackend.
# ----------------------------------------------------------------------


def _apply_action_call(backend: GymnasiumBackend, action: Any) -> Result:
    """Generic primitive: pass any action through ``backend.apply_action``."""
    try:
        out = backend.apply_action(action)
    except Exception as exc:  # noqa: BLE001
        return Result(status=ResultStatus.ERROR, message=f"{type(exc).__name__}: {exc}")
    return Result(
        status=ResultStatus.OK,
        observation=out,
        message=(
            f"step reward={out['reward']:.4g} "
            f"terminated={out['terminated']} truncated={out['truncated']}"
        ),
    )


def apply_action() -> Primitive:
    """Generic Gym action primitive. Used by VLAPolicy / scripted policies that
    emit raw action vectors instead of structured intents.
    """
    return Primitive(
        name="apply_action",
        call=_apply_action_call,
        description="Step the Gym environment by one action.",
        arg_schema={"action": "list[float] | int (env-specific shape)"},
    )


def _reset_call(backend: GymnasiumBackend) -> Result:
    out = backend.reset_env()
    return Result(
        status=ResultStatus.OK,
        observation=out,
        message="env reset",
    )


def reset_env() -> Primitive:
    """Reset the Gym env to its initial state. Useful for multi-episode benches."""
    return Primitive(
        name="reset_env",
        call=_reset_call,
        description="Reset the Gym environment to its initial state.",
        arg_schema={},
    )
