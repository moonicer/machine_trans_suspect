"""glossary.csv: loading and term matching.

Columns (header names are case-insensitive; only ru and kk are required):
  ru            source term, dictionary form («скин», «инвентарь», «семья»)
  kk            approved Kazakh term (stem / dictionary form)
  en            English term
  notes         usage notes for translators and reviewers
  kk_allowed    other acceptable renderings, ';'-separated (slang vs standard)
  kk_forbidden  renderings that violate the glossary, ';'-separated
  en_sense      regex that must match the English reference for the row to apply
                (disambiguates homographs: «пропуск» = pass / skip)
  strict        yes -> the term must appear whenever the source has it
                (brand-like game terms); a missing term is a finding

Multi-word terms are matched as substrings; single words as token stems with
Kazakh suffixes (see kk_morph).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import kk_morph

ALIASES = {"ru": ("ru", "rus", "russian", "рус"), "kk": ("kk", "kz", "kaz", "kazakh", "қаз"),
           "en": ("en", "eng", "english"), "notes": ("notes", "note", "comment", "usage", "usage_notes"),
           "kk_allowed": ("kk_allowed", "allowed", "kk_variants"),
           "kk_forbidden": ("kk_forbidden", "forbidden", "kk_wrong"),
           "en_sense": ("en_sense", "sense"), "strict": ("strict", "required")}

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]+(?:-[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]+)*")
RU_VOWEL_END = set("аяоеёьйиыую")


@dataclass
class Term:
    ru: str
    kk: str
    en: str = ""
    notes: str = ""
    allowed: list = field(default_factory=list)
    forbidden: list = field(default_factory=list)
    en_sense: str = ""
    strict: bool = False

    def __post_init__(self):
        self._ru_re = _ru_pattern(self.ru)

    def as_dict(self) -> dict:
        return {k: v for k, v in {"ru": self.ru, "kk": self.kk, "en": self.en, "notes": self.notes}.items() if v}

    # ---------- matching ----------
    def in_source(self, ru_text: str, en_text: str | None = None) -> bool:
        if not self._ru_re.search(ru_text):
            return False
        if self.en_sense and en_text is not None and not re.search(self.en_sense, en_text, re.I):
            return False
        return True

    def ok_forms(self) -> list[str]:
        return [self.kk] + self.allowed


def _ru_pattern(term: str) -> re.Pattern:
    """Russian term -> regex over inflected forms. Vowel/soft-sign final terms
    drop that letter and require an ending (семья -> семь+я/и/ю… but not «семь»);
    short stems only take noun endings (бан ≠ банк)."""
    t = term.lower().strip()
    if " " in t:
        return re.compile(r"(?<![а-яёa-z])" + re.escape(t), re.I)
    if t[-1] in RU_VOWEL_END and len(t) > 3:
        stem, ends = t[:-1], r"[а-яё]{1,4}"
    else:
        stem, ends = t, r"[а-яё]{0,4}"
    if len(stem) <= 3:
        ends = r"(?:а|у|ом|е|ы|ов|ам|ами|ах|и|ей|ю|я)?"
    return re.compile(r"(?<![а-яёa-z])" + re.escape(stem) + ends + r"(?![а-яё])", re.I)


def kk_find(text: str, form: str) -> list[tuple[int, int, str]]:
    """Occurrences of a Kazakh term (with suffixes) in text: (start, end, token)."""
    f = form.lower().strip()
    out = []
    if not f:
        return out
    if " " in f:
        for m in re.finditer(re.escape(f), text, re.I):
            out.append((m.start(), m.end(), text[m.start():m.end()]))
        return out
    exact = form.startswith("=")
    f = f.lstrip("=")
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.lower() == f:
            out.append((m.start(), m.end(), tok)); continue
        if exact or len(f) <= 3:
            # short stems: token must be stem + a regular nominal suffix chain
            if kk_morph.split_token(tok, f) is not None:
                out.append((m.start(), m.end(), tok))
        elif any(tok.lower().startswith(v) for v in kk_morph.stem_variants(f)):
            out.append((m.start(), m.end(), tok))
    return out


def _split(v: str) -> list[str]:
    return [x.strip() for x in (v or "").split(";") if x.strip()]


def load_terms(path: str | Path | None) -> list[Term]:
    if not path or not Path(path).exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    terms = []
    for r in rows:
        norm = {}
        for k, v in r.items():
            if k is None:
                continue
            for canon, names in ALIASES.items():
                if k.strip().lower() in names:
                    norm[canon] = (v or "").strip()
        if not norm.get("ru") or not norm.get("kk"):
            continue
        terms.append(Term(ru=norm["ru"], kk=norm["kk"], en=norm.get("en", ""), notes=norm.get("notes", ""),
                          allowed=_split(norm.get("kk_allowed", "")),
                          forbidden=_split(norm.get("kk_forbidden", "")),
                          en_sense=norm.get("en_sense", ""),
                          strict=norm.get("strict", "").lower() in ("yes", "y", "true", "1", "да", "иә")))
    return terms


# ---- backwards-compatible helpers used by the MT judge prompt ----
def load(path) -> list[dict]:
    return [t.as_dict() for t in load_terms(path)]


def relevant(glossary: list, source: str, target: str, limit: int = 6) -> list[dict]:
    out = []
    for g in glossary:
        t = g if isinstance(g, Term) else Term(ru=g.get("ru", "") or "-", kk=g.get("kk", ""),
                                               en=g.get("en", ""), notes=g.get("notes", ""))
        if (t.ru and t._ru_re.search(source)) or (t.kk and kk_find(target, t.kk)):
            out.append(t.as_dict())
    return out[:limit]
