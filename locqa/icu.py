"""Small ICU MessageFormat parser (the subset Flutter gen-l10n supports).

Used by the MT check to (a) analyse only human-readable text, never placeholder
names or plural keywords, and (b) validate LLM suggestions before they can
reach the report or the fixed ARB.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PH = "⟨X⟩"  # neutral marker inserted where a placeholder sits in running text


class IcuError(ValueError):
    pass


@dataclass
class Node:
    kind: str                      # "text" | "arg" | "plural" | "select"
    text: str = ""
    name: str = ""
    branches: dict = field(default_factory=dict)  # selector -> list[Node]


def _parse(s: str, i: int, depth: int) -> tuple[list[Node], int]:
    nodes: list[Node] = []
    buf: list[str] = []
    n = len(s)

    def flush():
        if buf:
            nodes.append(Node("text", text="".join(buf)))
            buf.clear()

    while i < n:
        c = s[i]
        if c == "'":
            # ICU quoting: '' -> ', '{...}' -> literal
            if i + 1 < n and s[i + 1] == "'":
                buf.append("'"); i += 2; continue
            if i + 1 < n and s[i + 1] in "{}#|":
                j = s.find("'", i + 1)
                if j == -1:
                    buf.append(s[i + 1:]); i = n; continue
                buf.append(s[i + 1:j]); i = j + 1; continue
            buf.append(c); i += 1; continue
        if c == "{":
            flush()
            node, i = _parse_arg(s, i + 1, depth)
            nodes.append(node)
            continue
        if c == "}":
            if depth == 0:
                raise IcuError(f"unbalanced '}}' at {i}")
            flush()
            return nodes, i
        buf.append(c); i += 1
    if depth > 0:
        raise IcuError("unterminated '{'")
    flush()
    return nodes, i


_ARG_HEAD = re.compile(r"\s*([A-Za-z_][\w]*)\s*(?:,\s*(plural|select|selectordinal)\s*,)?\s*")


def _parse_arg(s: str, i: int, depth: int) -> tuple[Node, int]:
    m = _ARG_HEAD.match(s, i)
    if not m or not m.group(1):
        raise IcuError(f"bad argument at {i}")
    name, typ = m.group(1), m.group(2)
    i = m.end()
    if typ is None:
        # simple {name} (also tolerate {name, number} style formats)
        j = s.find("}", i)
        if j == -1:
            raise IcuError("unterminated argument")
        if s[i:j].strip() and not s[i:j].lstrip().startswith(","):
            raise IcuError(f"junk in argument {name!r}")
        return Node("arg", name=name), j + 1
    node = Node("plural" if typ != "select" else "select", name=name)
    n = len(s)
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i < n and s[i] == "}":
            if "other" not in node.branches:
                raise IcuError(f"{typ} '{name}' has no 'other' branch")
            return node, i + 1
        m2 = re.compile(r"(offset:\s*\d+|=\d+|[A-Za-z_]\w*)\s*").match(s, i)
        if not m2:
            raise IcuError(f"bad selector in {name!r} at {i}")
        sel = m2.group(1)
        i = m2.end()
        if sel.startswith("offset:"):
            continue
        if i >= n or s[i] != "{":
            raise IcuError(f"selector {sel!r} without body")
        body, i = _parse(s, i + 1, depth + 1)
        if i >= n or s[i] != "}":
            raise IcuError("unterminated branch")
        node.branches[sel] = body
        i += 1
    raise IcuError(f"unterminated {typ} '{name}'")


def parse(s: str) -> list[Node]:
    nodes, _ = _parse(s, 0, 0)
    return nodes


def placeholders(s: str) -> set[str]:
    out: set[str] = set()

    def walk(nodes):
        for nd in nodes:
            if nd.kind != "text":
                out.add(nd.name)
            for b in nd.branches.values():
                walk(b)
    walk(parse(s))
    return out


def text_segments(s: str) -> list[str]:
    """Readable text variants: one per plural/select branch combination
    (flattened: each branch body is rendered with sibling text). Placeholders
    become PH so that analysers never mistake names for words."""
    try:
        nodes = parse(s)
    except IcuError:
        return [re.sub(r"\{[^{}]*\}", PH, s)]

    def render(nodes) -> list[str]:
        variants = [""]
        for nd in nodes:
            if nd.kind == "text":
                variants = [v + nd.text.replace("#", PH) for v in variants]
            elif nd.kind == "arg":
                variants = [v + PH for v in variants]
            else:
                opts = []
                for body in nd.branches.values():
                    opts.extend(render(body))
                opts = opts[:6] or [""]
                variants = [v + o for v in variants for o in opts][:12]
        return variants

    seen, out = set(), []
    for v in render(nodes):
        v = v.strip()
        if v and v not in seen:
            seen.add(v); out.append(v)
    return out or [""]


def is_valid(s: str) -> bool:
    try:
        parse(s)
        return True
    except IcuError:
        return False


def same_structure(a: str, b: str) -> bool:
    """True if b keeps a's placeholder set and plural/select argument names."""
    try:
        return placeholders(a) == placeholders(b)
    except IcuError:
        return False
