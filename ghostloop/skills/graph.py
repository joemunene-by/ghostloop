"""SkillGraph — typed DAG over Primitives + composite primitives.

Used for:

  - Mission feasibility checks: given a mission's required skills,
    every prerequisite is satisfied.
  - Cross-embodiment reuse: a skill name like "pick" is implemented
    differently per morphology, but the GRAPH stays the same.
  - Documentation: render the dependency tree as Markdown for the
    operator-facing docs page.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..core import Primitive


class SkillError(ValueError):
    """Raised on cycle / unknown-prereq / duplicate-name errors."""


@dataclass
class Skill:
    """One node in the skill graph."""

    name: str
    primitive: Primitive
    prerequisites: list[str] = field(default_factory=list)
    refines: str | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def skill_from_primitive(
    primitive: Primitive,
    *,
    prerequisites: list[str] | None = None,
    refines: str | None = None,
    description: str = "",
) -> Skill:
    """Convenience: turn a Primitive into a Skill, inheriting its name."""
    return Skill(
        name=primitive.name,
        primitive=primitive,
        prerequisites=list(prerequisites or []),
        refines=refines,
        description=description or primitive.description,
    )


@dataclass
class SkillGraph:
    """A typed DAG of Skills with prereq + refinement edges."""

    skills: dict[str, Skill] = field(default_factory=dict)

    def add(self, skill: Skill) -> None:
        if skill.name in self.skills:
            raise SkillError(f"duplicate skill name {skill.name!r}")
        self.skills[skill.name] = skill

    def add_many(self, skills: list[Skill]) -> None:
        for s in skills:
            self.add(s)

    def get(self, name: str) -> Skill:
        if name not in self.skills:
            raise SkillError(f"unknown skill {name!r}")
        return self.skills[name]

    def validate(self) -> None:
        """Check for unknown prereqs, unknown refines targets, and cycles.

        Raises SkillError on the first problem found.
        """
        for s in self.skills.values():
            for dep in s.prerequisites:
                if dep not in self.skills:
                    raise SkillError(
                        f"skill {s.name!r} depends on unknown {dep!r}"
                    )
            if s.refines is not None and s.refines not in self.skills:
                raise SkillError(
                    f"skill {s.name!r} refines unknown {s.refines!r}"
                )
        # Cycle detection via Kahn's. Edge: dep -> dependent.
        indeg = {name: 0 for name in self.skills}
        out_edges: dict[str, list[str]] = {n: [] for n in self.skills}
        for s in self.skills.values():
            for dep in s.prerequisites:
                indeg[s.name] += 1
                out_edges[dep].append(s.name)
        ready = deque([n for n, d in indeg.items() if d == 0])
        seen = 0
        while ready:
            n = ready.popleft()
            seen += 1
            for nxt in out_edges[n]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
        if seen != len(self.skills):
            raise SkillError("dependency cycle in skill graph")

    def topological_order(self) -> list[str]:
        """Return skills in a valid topological order. Raises if cyclic."""
        self.validate()
        order: list[str] = []
        indeg = {n: len(s.prerequisites) for n, s in self.skills.items()}
        out_edges: dict[str, list[str]] = {n: [] for n in self.skills}
        for s in self.skills.values():
            for dep in s.prerequisites:
                out_edges[dep].append(s.name)
        ready = deque(
            sorted(n for n, d in indeg.items() if d == 0)
        )
        while ready:
            n = ready.popleft()
            order.append(n)
            for nxt in sorted(out_edges[n]):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
        return order

    def required_for(self, skill_name: str) -> set[str]:
        """All transitive prerequisites of ``skill_name`` (excluding itself)."""
        if skill_name not in self.skills:
            raise SkillError(f"unknown skill {skill_name!r}")
        result: set[str] = set()
        stack = list(self.skills[skill_name].prerequisites)
        while stack:
            n = stack.pop()
            if n in result:
                continue
            if n not in self.skills:
                raise SkillError(f"unknown prereq {n!r} for {skill_name!r}")
            result.add(n)
            stack.extend(self.skills[n].prerequisites)
        return result

    def descendants(self, skill_name: str) -> set[str]:
        """All transitive dependents of ``skill_name`` (skills that need it)."""
        if skill_name not in self.skills:
            raise SkillError(f"unknown skill {skill_name!r}")
        # Build reverse edges.
        reverse: dict[str, set[str]] = {n: set() for n in self.skills}
        for s in self.skills.values():
            for dep in s.prerequisites:
                reverse[dep].add(s.name)
        result: set[str] = set()
        stack = list(reverse.get(skill_name, set()))
        while stack:
            n = stack.pop()
            if n in result:
                continue
            result.add(n)
            stack.extend(reverse.get(n, set()))
        return result

    def coverage(self, required: list[str]) -> dict[str, Any]:
        """Check whether the graph covers a set of required skills.

        Returns a dict carrying ``covered`` (list of names present + their
        prereqs in the graph), ``missing`` (required names not in the
        graph), ``transitive_deps`` (every prereq that needs to be
        present for the required set to work).
        """
        present = [n for n in required if n in self.skills]
        missing = [n for n in required if n not in self.skills]
        transitive: set[str] = set()
        for n in present:
            transitive |= self.required_for(n)
        return {
            "n_required": len(required),
            "n_covered": len(present),
            "covered": present,
            "missing": missing,
            "transitive_deps": sorted(transitive),
        }

    def render_md(self) -> str:
        order = self.topological_order()
        lines = ["# Skill graph", ""]
        for name in order:
            s = self.skills[name]
            prereq = ", ".join(s.prerequisites) if s.prerequisites else "(none)"
            refine = f" — refines `{s.refines}`" if s.refines else ""
            desc = f" — {s.description}" if s.description else ""
            lines.append(f"- **{name}**: prereqs={prereq}{refine}{desc}")
        return "\n".join(lines) + "\n"
