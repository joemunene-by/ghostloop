"""Teacher-student distillation for embodied policies.

A teacher policy (typically an LLM-driven loop or a slow but capable
VLA) generates rollouts under the safety pipeline; a student policy
trains on the resulting (observation, action) pairs to imitate the
teacher faster + cheaper. Standard distillation, made explicit here
because ghostloop's typed Trace makes the data-collection side trivial.

Three components:

  - ``DistillationDataset`` — collects (state_before, intent) pairs
    from a list of Rollout objects (or from raw Traces). Filters to
    only keep events whose decision was ALLOW so the student doesn't
    learn from blocked actions. Optional include_denied flag for users
    who want to teach the student to *anticipate* denials.
  - ``StudentPolicy`` Protocol — same shape as PolicyAdapter from the
    safe-RL harness. Plug in PyTorch / JAX / a from-scratch numpy MLP /
    a small VLA head — anything with ``act(obs)`` + ``update(batch)``.
  - ``train_distill(...)`` — the outer loop. Iterates over the dataset
    in batches, calls ``student.update(batch)`` per batch, optionally
    runs a held-out eval rollout every N iterations.

Pure-stdlib + the existing rollout types. The student framework is
your problem; we provide the data shape and the loop.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from ..core import Intent, Trace, TraceEvent
from .core import Rollout, Transition


@dataclass
class DistillationSample:
    """One supervised example: state -> intent.

    ``intent_args`` keeps the dict shape so a student can learn either
    discrete primitive names OR continuous arg vectors (depending on
    its action head).
    """

    state: dict[str, Any]
    intent_name: str
    intent_args: dict[str, Any]
    decision_action: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "intent_name": self.intent_name,
            "intent_args": self.intent_args,
            "decision_action": self.decision_action,
            "metadata": self.metadata,
        }


@dataclass
class DistillationDataset:
    """Collected (state, intent) pairs ready for student training."""

    samples: list[DistillationSample] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.samples)

    def add_trace(
        self, trace: Trace,
        *,
        include_denied: bool = False,
        include_errors: bool = False,
    ) -> int:
        """Append every event of a Trace as a sample. Returns count added.

        ``include_denied`` keeps events the safety pipeline blocked,
        flagged via ``metadata['was_denied']``. Useful when teaching the
        student to *avoid* denials by example.

        ``include_errors`` keeps events whose runtime call returned an
        ERROR result. Default off; the student usually shouldn't imitate
        crashes.
        """
        added = 0
        for ev in trace.events:
            decision_action = ev.decision.action.value
            if not include_denied and decision_action == "deny":
                continue
            if not include_errors and ev.result.status.value == "error":
                continue
            self.samples.append(DistillationSample(
                state=dict(ev.state_before) if isinstance(ev.state_before, dict) else {},
                intent_name=ev.intent.name,
                intent_args=dict(ev.intent.args),
                decision_action=decision_action,
                metadata={
                    "was_denied": decision_action == "deny",
                    "result_status": ev.result.status.value,
                    "timestamp": ev.timestamp,
                    "step": ev.step,
                    "episode_id": trace.episode_id,
                },
            ))
            added += 1
        return added

    def add_rollout(
        self, rollout: Rollout,
        *,
        include_denied: bool = False,
    ) -> int:
        """Append a Rollout's transitions. Each transition becomes one sample."""
        added = 0
        for t in rollout.transitions:
            if not include_denied and t.violated:
                continue
            obs = t.obs if isinstance(t.obs, dict) else {"obs": t.obs}
            args = t.action if isinstance(t.action, dict) else {"action": t.action}
            self.samples.append(DistillationSample(
                state=obs,
                intent_name=t.intent_name,
                intent_args=args,
                decision_action="deny" if t.violated else "allow",
                metadata={
                    "was_denied": t.violated,
                    "reward": t.reward,
                    "done": t.done,
                },
            ))
            added += 1
        return added

    def shuffle(self, seed: int = 0) -> None:
        rng = random.Random(seed)
        rng.shuffle(self.samples)

    def split(self, train_frac: float = 0.9) -> tuple["DistillationDataset", "DistillationDataset"]:
        cut = int(self.n * train_frac)
        train = DistillationDataset(samples=self.samples[:cut])
        val = DistillationDataset(samples=self.samples[cut:])
        return train, val

    def batches(self, batch_size: int = 32) -> Iterable[list[DistillationSample]]:
        for i in range(0, self.n, batch_size):
            yield self.samples[i : i + batch_size]

    def filter_by_primitive(self, names: list[str]) -> "DistillationDataset":
        keep = set(names)
        return DistillationDataset(
            samples=[s for s in self.samples if s.intent_name in keep],
        )

    def primitive_distribution(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.samples:
            out[s.intent_name] = out.get(s.intent_name, 0) + 1
        return out

    def write_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w") as f:
            for s in self.samples:
                f.write(json.dumps(s.to_json()) + "\n")

    @classmethod
    def read_jsonl(cls, path: str | Path) -> "DistillationDataset":
        path = Path(path)
        samples: list[DistillationSample] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                samples.append(DistillationSample(
                    state=obj.get("state", {}),
                    intent_name=obj["intent_name"],
                    intent_args=obj.get("intent_args", {}),
                    decision_action=obj.get("decision_action", "allow"),
                    metadata=obj.get("metadata", {}),
                ))
        return cls(samples=samples)


# ---------------------------------------------------------------------------
# Student protocol + training loop.
# ---------------------------------------------------------------------------


class StudentPolicy(Protocol):
    """Anything with these two methods is a valid student.

    The framework is yours: PyTorch nn.Module, JAX flax model, scripted
    nearest-neighbour, even a tabular policy. The contract is:

      ``act(state)`` — given a state, return an Intent.
      ``update(batch)`` — given a batch of DistillationSamples, do one
        gradient / regression / fitting step. Return a metrics dict.
    """

    def act(self, state: dict[str, Any]) -> Intent: ...

    def update(self, batch: list[DistillationSample]) -> dict[str, float]: ...


@dataclass
class DistillationTrainer:
    """Outer loop for student training over a DistillationDataset."""

    student: StudentPolicy
    dataset: DistillationDataset
    val_dataset: DistillationDataset | None = None
    batch_size: int = 32
    n_epochs: int = 10
    log_every: int = 1
    seed: int = 0
    on_epoch: Callable[[int, dict[str, Any]], None] | None = None

    def evaluate(self, ds: DistillationDataset) -> dict[str, float]:
        """Compute per-primitive accuracy of the student on a held-out set.

        Defines accuracy as: predicted intent name matches the teacher's,
        regardless of args. For continuous-action distillation this is a
        weak signal; override by passing your own callback or computing
        action-MSE inside the student.
        """
        if not ds.samples:
            return {"accuracy": 0.0, "n": 0}
        correct = 0
        for s in ds.samples:
            try:
                pred = self.student.act(s.state)
            except Exception:  # noqa: BLE001
                continue
            if pred.name == s.intent_name:
                correct += 1
        return {"accuracy": correct / ds.n, "n": ds.n}

    def train(self) -> list[dict[str, Any]]:
        rng = random.Random(self.seed)
        history: list[dict[str, Any]] = []
        for epoch in range(self.n_epochs):
            started = time.time()
            self.dataset.shuffle(seed=self.seed + epoch)
            epoch_metrics: dict[str, float] = {}
            n_batches = 0
            for batch in self.dataset.batches(self.batch_size):
                update_metrics = self.student.update(batch)
                for k, v in update_metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + float(v)
                n_batches += 1
            for k in list(epoch_metrics.keys()):
                epoch_metrics[k] /= max(1, n_batches)
            record: dict[str, Any] = {
                "epoch": epoch,
                "duration_s": round(time.time() - started, 4),
                "n_samples": self.dataset.n,
                "n_batches": n_batches,
                **{f"train/{k}": v for k, v in epoch_metrics.items()},
            }
            if self.val_dataset is not None and self.val_dataset.n > 0:
                val_metrics = self.evaluate(self.val_dataset)
                record.update({f"val/{k}": v for k, v in val_metrics.items()})
            history.append(record)
            if self.on_epoch is not None:
                self.on_epoch(epoch, record)
            elif self.log_every > 0 and epoch % self.log_every == 0:
                acc = record.get("val/accuracy")
                acc_str = f" val_acc={acc:.3f}" if acc is not None else ""
                print(
                    f"[distill] epoch={epoch} batches={n_batches} "
                    f"duration={record['duration_s']}s{acc_str}"
                )
        return history


def collect_dataset(
    teacher_run: Callable[[], Trace | Rollout | list[Trace] | list[Rollout]],
    *,
    include_denied: bool = False,
    include_errors: bool = False,
) -> DistillationDataset:
    """Convenience: run a teacher callable + harvest the resulting traces.

    ``teacher_run`` is expected to return a Trace, a Rollout, or a list
    of either. Useful for one-line dataset construction:

        ds = collect_dataset(lambda: my_teacher_loop(runtime, n_episodes=20))
    """
    out = DistillationDataset()
    result = teacher_run()
    items: list = result if isinstance(result, list) else [result]
    for item in items:
        if isinstance(item, Trace):
            out.add_trace(item, include_denied=include_denied, include_errors=include_errors)
        elif isinstance(item, Rollout):
            out.add_rollout(item, include_denied=include_denied)
    return out
