"""Build ``intl_kk_fixed.arb`` from a reviewed report.

The target file is edited *line by line*, never re-serialised, because real ARB files
contain repeated ``"@@comment"`` markers, duplicate keys (last one wins) and
hand-written formatting that a JSON round trip would destroy.

For every changed string a metadata entry is written next to it::

    "key": "new text",
    "@key": {"x-locqa-reviewed": true, "x-locqa-issue": "machine_translation_suspect", ...}

Flutter's ``gen-l10n`` reads resource attributes from the template ARB only, so extra
``x-`` attributes in a non-template locale file are inert. If the file already has an
``@key`` object, the flags are merged into it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import icu
from .arb import ArbFile

_KEY_RE = r'^(\s*)"{key}"\s*:\s*'
_ANY_KEY = re.compile(r'^(\s*)"((?:[^"\\]|\\.)*)"\s*:\s*(.*)$')


def select(findings: list[dict], include_unreviewed: bool = False) -> dict[str, dict]:
    """key -> {"text", "reviewed", "issues", "reviewer", "note"}; one final string per key."""
    out: dict[str, dict] = {}
    for f in findings:
        text = f.get("suggestion")
        reviewed = bool(f.get("reviewed_by_human"))
        if not text or (not reviewed and not include_unreviewed):
            continue
        rev = (f.get("details") or {}).get("review") or {}
        cur = out.get(f["key"])
        if cur is None:
            out[f["key"]] = {"text": text, "reviewed": reviewed, "issues": [f["issue_type"]],
                             "reviewer": rev.get("reviewer"), "note": rev.get("note")}
            continue
        if cur["text"] != text:
            raise SystemExit(f"{f['key']}: two findings propose different final text; "
                             "resolve it in review (one string must fix every issue of the key)")
        cur["issues"].append(f["issue_type"])
        cur["reviewed"] = cur["reviewed"] and reviewed
        cur["reviewer"] = cur["reviewer"] or rev.get("reviewer")
    return out


def _encode(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _meta_block(indent: str, key: str, info: dict, existing: dict | None) -> list[str]:
    flags = {"x-locqa-reviewed": info["reviewed"],
             "x-locqa-issue": ",".join(dict.fromkeys(info["issues"]))}
    if info.get("reviewer"):
        flags["x-locqa-reviewer"] = info["reviewer"]
    if existing is None:
        return [f'{indent}"@{key}": {json.dumps(flags, ensure_ascii=False)}']
    merged = dict(existing)
    merged.update(flags)
    body = json.dumps(merged, ensure_ascii=False, indent=2).split("\n")
    return [f'{indent}"@{key}": {body[0]}'] + [indent + b for b in body[1:]]


def _find_meta(lines: list[str], key: str) -> tuple[int, int, dict] | None:
    """(first_line, last_line, parsed) of an existing ``"@key": {...}`` object, if any."""
    pat = re.compile(_KEY_RE.format(key=re.escape("@" + key)))
    for i, l in enumerate(lines):
        m = pat.match(l)
        if not m:
            continue
        text = "\n".join(lines[i:])
        start = m.end()
        try:
            obj, end = json.JSONDecoder().raw_decode(text[start:])
        except ValueError:
            return None
        consumed = text[:start + end]
        return i, i + consumed.count("\n"), obj
    return None


def apply_patch(target_text: str, changes: dict[str, dict], source: ArbFile | None = None):
    """Return (new_text, stats). ``changes`` comes from :func:`select`."""
    lines = target_text.split("\n")
    stats = {"changed": 0, "reviewed": 0, "unreviewed": 0, "duplicate_lines_patched": 0,
             "missing_keys": [], "skipped": []}
    todo = {}
    for key, info in changes.items():
        if source is not None:
            src = source.get(key)
            if src is None:
                stats["skipped"].append((key, "key not in source"))
                continue
            if not icu.is_valid(info["text"]) or not icu.same_structure(src.value, info["text"]):
                raise SystemExit(f"{key}: fix breaks ICU syntax or placeholders of the source")
        elif not icu.is_valid(info["text"]):
            raise SystemExit(f"{key}: fix is not valid ICU")
        todo[key] = info

    # locate every line that defines a wanted key (duplicates: patch all, flag the last = effective one)
    occ: dict[str, list[int]] = {}
    for i, l in enumerate(lines):
        m = _ANY_KEY.match(l)
        if m and m.group(2) in todo:
            occ.setdefault(m.group(2), []).append(i)
    for key in todo:
        if key not in occ:
            stats["missing_keys"].append(key)

    # decide the edits first (on the original line numbers), then apply bottom-up
    edits = []  # (line_idx, kind, payload)
    for key, idxs in occ.items():
        info = todo[key]
        for i in idxs:
            l = lines[i]
            m = _ANY_KEY.match(l)
            indent, rest = m.group(1), m.group(3)
            comma = rest.rstrip().endswith(",")
            prefix = l[:len(l) - len(rest)]
            edits.append((i, "value", prefix + _encode(info["text"]) + ("," if comma else "")))
            if i != idxs[-1]:
                stats["duplicate_lines_patched"] += 1
        last = idxs[-1]
        indent = _ANY_KEY.match(lines[last]).group(1)
        comma = _ANY_KEY.match(lines[last]).group(3).rstrip().endswith(",")
        meta = _find_meta(lines, key)
        if meta:
            a, b, obj = meta
            edits.append((a, "meta_replace", (b, _meta_block(indent, key, info, obj),
                                              lines[b].rstrip().endswith(","))))
        else:
            edits.append((last, "meta_insert", (_meta_block(indent, key, info, None), comma)))
        stats["changed"] += 1
        stats["reviewed" if info["reviewed"] else "unreviewed"] += 1

    # Apply from the bottom up so earlier line numbers stay valid.
    for i, kind, payload in sorted(edits, key=lambda e: (e[0], e[1] != "meta_insert"), reverse=True):
        if kind == "value":
            lines[i] = payload
        elif kind == "meta_insert":
            block, comma = payload
            if not comma:  # the changed line was the last entry: it now needs a comma
                lines[i] = lines[i].rstrip() + ","
            block[-1] += "," if comma else ""
            lines[i + 1:i + 1] = block
        else:
            end, block, comma = payload
            block[-1] += "," if comma else ""
            lines[i:end + 1] = block
    return "\n".join(lines), stats


def validate_output(new_text: str, source: ArbFile, changed: set[str]) -> list[str]:
    """Problems that would break the Flutter build; an empty list means the file is sound."""
    problems = []
    try:
        data = json.loads(new_text)
    except ValueError as e:
        return [f"not valid JSON: {e}"]
    for k in changed:
        v, s = data.get(k), source.get(k)
        if not isinstance(v, str):
            problems.append(f"{k}: missing after patch")
        elif s and (not icu.is_valid(v) or not icu.same_structure(s.value, v)):
            problems.append(f"{k}: placeholder/ICU mismatch with source")
    return problems


def build(target: str, source: str, report: str, out: str, include_unreviewed: bool = False) -> dict:
    rep = json.loads(Path(report).read_text("utf-8"))
    findings = rep if isinstance(rep, list) else rep.get("findings", [])
    changes = select(findings, include_unreviewed)
    text = Path(target).read_text("utf-8")
    src = ArbFile(source)
    new_text, stats = apply_patch(text, changes, src)
    problems = validate_output(new_text, src, set(changes) - set(stats["missing_keys"]))
    if problems:
        raise SystemExit("refusing to write intl_kk_fixed.arb:\n  " + "\n  ".join(problems[:20]))
    Path(out).write_text(new_text, "utf-8")
    side = Path(out).with_suffix(".review.json")
    old = ArbFile(target)
    side.write_text(json.dumps(
        {k: {"before": (old.get(k).value if old.get(k) else None), "after": v["text"],
             "issues": list(dict.fromkeys(v["issues"])), "reviewed_by_human": v["reviewed"],
             "reviewer": v["reviewer"], "note": v["note"]} for k, v in changes.items()},
        ensure_ascii=False, indent=2), "utf-8")
    stats["findings_without_fix"] = sorted({f["key"] for f in findings if not f.get("suggestion")})
    return stats
