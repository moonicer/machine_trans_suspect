"""glossary.csv loader tolerant to header naming (ru/RU/russian, kk/kaz, en, notes)."""
from __future__ import annotations

import csv
from pathlib import Path

ALIASES = {"ru": ("ru", "rus", "russian", "рус"), "kk": ("kk", "kz", "kaz", "kazakh", "қаз"),
           "en": ("en", "eng", "english"), "notes": ("notes", "note", "comment", "usage", "usage_notes")}


def load(path: str | Path | None) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        norm = {}
        for k, v in r.items():
            if k is None:
                continue
            lk = k.strip().lower()
            for canon, names in ALIASES.items():
                if lk in names:
                    norm[canon] = (v or "").strip()
        if norm.get("ru") or norm.get("kk"):
            out.append(norm)
    return out


def relevant(glossary: list[dict], source: str, target: str, limit: int = 6) -> list[dict]:
    s, t = source.lower(), target.lower()
    hits = []
    for g in glossary:
        ru = g.get("ru", "").lower()
        kk = g.get("kk", "").lower()
        stem = ru[:max(4, len(ru) - 2)] if len(ru) > 4 else ru  # crude RU stemming
        if (ru and stem and stem in s) or (kk and kk in t):
            hits.append({k: v for k, v in g.items() if v})
    return hits[:limit]
