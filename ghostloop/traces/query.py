"""Trace query DSL — filter trace events with a small expression language.

Once a deployment runs for a week the JSONL trace files become a few
gigabytes. Grep is fine for one or two filters; once you're chaining
("any deny by the geofence gate, on a move_to intent, where x > 0.5")
you want a real query language.

This module provides a tiny single-line expression DSL that runs over a
``Trace`` (or any iterable of TraceEvents). The language is deliberately
narrow: comparison operators on dotted attribute paths plus ``and``,
``or``, ``not``, parentheses, and string / numeric literals.

Examples:

    intent.name == "move_to"
    decision.action == "deny" and intent.name == "move_to"
    result.status == "error" and intent.args.x > 0.5
    not (intent.name == "scan")
    intent.name in ("move_to", "scan", "pick")

Pure stdlib: tokenises with ``re``, parses recursive-descent, evaluates
against each event. No eval / exec — every operator is dispatched
through a fixed table so the DSL is safe to expose over HTTP.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..core import Trace, TraceEvent


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


_TOKEN_SPEC = [
    ("NUMBER",   r"-?\d+(?:\.\d+)?"),
    ("STRING",   r"\"[^\"]*\"|'[^']*'"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("COMMA",    r","),
    ("OP",       r"==|!=|>=|<=|>|<|\b(?:and|or|not|in)\b"),
    ("IDENT",    r"[a-zA-Z_][a-zA-Z0-9_\.]*"),
    ("WS",       r"\s+"),
]
_TOKEN_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOKEN_SPEC))


def _tokenise(expr: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise QueryError(f"unexpected character at offset {pos}: {expr[pos]!r}")
        kind = m.lastgroup
        value = m.group()
        pos = m.end()
        if kind == "WS":
            continue
        tokens.append((kind, value))
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------


class QueryError(ValueError):
    """Raised on parse / runtime errors during query evaluation."""


@dataclass
class _Parser:
    tokens: list[tuple[str, str]]
    i: int = 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def eat(self, kind: str) -> tuple[str, str]:
        tok = self.peek()
        if tok is None or tok[0] != kind:
            raise QueryError(f"expected {kind} but got {tok}")
        self.i += 1
        return tok

    def maybe(self, kind: str, value: str | None = None) -> tuple[str, str] | None:
        tok = self.peek()
        if tok is None or tok[0] != kind:
            return None
        if value is not None and tok[1] != value:
            return None
        self.i += 1
        return tok

    def parse_expr(self):
        return self.parse_or()

    def parse_or(self):
        left = self.parse_and()
        while self.maybe("OP", "or"):
            right = self.parse_and()
            left = ("or", left, right)
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.maybe("OP", "and"):
            right = self.parse_not()
            left = ("and", left, right)
        return left

    def parse_not(self):
        if self.maybe("OP", "not"):
            inner = self.parse_not()
            return ("not", inner)
        return self.parse_compare()

    def parse_compare(self):
        left = self.parse_atom()
        tok = self.peek()
        if tok and tok[0] == "OP" and tok[1] in ("==", "!=", ">=", "<=", ">", "<", "in"):
            op = tok[1]
            self.i += 1
            right = self.parse_atom()
            return ("cmp", op, left, right)
        return left

    def parse_atom(self):
        tok = self.peek()
        if tok is None:
            raise QueryError("unexpected end of expression")
        if tok[0] == "LPAREN":
            self.i += 1
            # Tuple-literal vs sub-expression.
            saved = self.i
            try:
                inner = self.parse_expr()
            except QueryError:
                self.i = saved
                inner = None
            if self.maybe("COMMA"):
                values = [inner] if inner is not None else []
                while not self.maybe("RPAREN"):
                    item = self.parse_atom()
                    values.append(item)
                    if not self.maybe("COMMA"):
                        self.eat("RPAREN")
                        break
                return ("tuple", values)
            self.eat("RPAREN")
            return inner
        if tok[0] == "NUMBER":
            self.i += 1
            value = float(tok[1]) if "." in tok[1] else int(tok[1])
            return ("lit", value)
        if tok[0] == "STRING":
            self.i += 1
            return ("lit", tok[1][1:-1])
        if tok[0] == "IDENT":
            self.i += 1
            return ("path", tok[1])
        raise QueryError(f"unexpected token {tok}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _resolve_path(event: TraceEvent, path: str) -> Any:
    """Resolve dotted path against a TraceEvent. Returns None for missing keys."""
    parts = path.split(".")
    obj: Any = event
    for p in parts:
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj = obj.get(p)
            continue
        # Try attribute, then JSON-style get on the to_json output.
        if hasattr(obj, p):
            obj = getattr(obj, p)
            continue
        if hasattr(obj, "to_json"):
            try:
                obj = obj.to_json().get(p)
                continue
            except (AttributeError, TypeError):
                pass
        return None
    # If the final value is an enum-like with .value, unwrap.
    if hasattr(obj, "value") and not isinstance(obj, (dict, list, tuple, str, int, float, bool)):
        try:
            return obj.value
        except Exception:  # noqa: BLE001
            return obj
    return obj


def _eval_node(node, event: TraceEvent) -> Any:
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "path":
        return _resolve_path(event, node[1])
    if kind == "tuple":
        return tuple(_eval_node(c, event) for c in node[1])
    if kind == "not":
        return not _eval_node(node[1], event)
    if kind == "and":
        return _eval_node(node[1], event) and _eval_node(node[2], event)
    if kind == "or":
        return _eval_node(node[1], event) or _eval_node(node[2], event)
    if kind == "cmp":
        op, left, right = node[1], node[2], node[3]
        lv = _eval_node(left, event)
        rv = _eval_node(right, event)
        try:
            if op == "==":
                return lv == rv
            if op == "!=":
                return lv != rv
            if op == "in":
                return lv in rv if rv is not None else False
            if op == ">":
                return lv > rv
            if op == "<":
                return lv < rv
            if op == ">=":
                return lv >= rv
            if op == "<=":
                return lv <= rv
        except TypeError:
            return False
    raise QueryError(f"unhandled node {node}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_query(expr: str):
    """Parse a query expression and return a callable predicate over events.

    The returned callable takes a TraceEvent and returns True / False
    according to the parsed expression. Use this if you want to filter
    a stream of events incrementally instead of materialising a Trace.
    """
    parser = _Parser(_tokenise(expr))
    ast = parser.parse_expr()
    if parser.i != len(parser.tokens):
        raise QueryError(
            f"unexpected trailing input at offset {parser.i}: {parser.tokens[parser.i:]}"
        )

    def predicate(event: TraceEvent) -> bool:
        try:
            return bool(_eval_node(ast, event))
        except (TypeError, ValueError):
            return False

    return predicate


def query(trace: Trace | Iterable[TraceEvent], expr: str) -> list[TraceEvent]:
    """Filter a Trace (or any iterable of TraceEvents) with a query expression.

    Returns a new list of events for which the expression evaluates
    truthy. Pure function: no side effects on the trace.
    """
    pred = compile_query(expr)
    if isinstance(trace, Trace):
        events = trace.events
    else:
        events = list(trace)
    return [ev for ev in events if pred(ev)]
