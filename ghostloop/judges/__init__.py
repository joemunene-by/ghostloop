"""Trace judges — score recorded episodes against rubrics.

Two judge styles ship in v0.10:

  - LLMJudge: pass a Trace through an OpenAI-compatible chat client,
    receive a structured rubric score (pass/fail/marginal + reasons +
    per-criterion scores). Free reward-modelling-quality eval over
    your full trace history; pairs naturally with GhostLM.
  - HeuristicJudge: stdlib rule-based scoring for environments where
    you can't (or won't) call an LLM. Configurable rubric of
    Predicate -> weight; sums to a final score.
"""

from .heuristic import HeuristicJudge, JudgeRule, RubricScore
from .llm_judge import LLMJudge, LLMJudgeConfig, LLMJudgement, default_rubric_prompt

__all__ = [
    "HeuristicJudge",
    "JudgeRule",
    "RubricScore",
    "LLMJudge",
    "LLMJudgeConfig",
    "LLMJudgement",
    "default_rubric_prompt",
]
