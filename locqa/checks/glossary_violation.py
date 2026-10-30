"""glossary_violation: the target conflicts with the agreed game glossary.

For every glossary row whose Russian term occurs in the source (and, if the
row has `en_sense`, whose English sense matches the EN reference):
  * forbidden rendering present          -> violation (high); the fix swaps the
    stem and rebuilds the Kazakh endings (терісін -> скинін)
  * forbidden present but the approved term is there too -> not reported here
    (common words like «сыйлық» co-occur legitimately; inconsistent_term owns it)
  * strict term absent from a short label -> term missing (low, no auto-fix);
    long sentences paraphrase legitimately, so they are not checked for absence
One finding per key; all term problems of that key are combined.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import icu, kk_morph
from ..arb import ArbFile
from ..finding import Finding
from ..glossary import Term, kk_find
from ..text_kk import language_profile

ISSUE = "glossary_violation"


@dataclass
class GlossaryConfig:
    report_missing: bool = True        # strict terms absent from the target
    missing_confidence: float = 0.55
    missing_max_words: int = 6         # only labels/titles are checked for absence


def _eligible(src: str, tgt: str) -> bool:
    if not tgt.strip() or tgt.strip() == src.strip():
        return False                     # untranslated / same_as_source own these
    prof = language_profile(tgt)
    if prof["n_cyr"] == 0:
        return False                     # Latin-only (e.g. Uzbek): wrong_language
    return prof["ru_share"] <= 0.5


def _fix(target: str, bad: list[tuple[int, int, str, str]], approved: str) -> tuple[str | None, list[str]]:
    """Replace each (start, end, token, bad_form) right-to-left. Returns
    (new_text or None, notes)."""
    notes, out = [], target
    for start, end, tok, form in sorted(bad, key=lambda x: -x[0]):
        form = form.lstrip("=")
        if " " in form:
            new = approved if tok.islower() or not tok[:1].isupper() else approved[:1].upper() + approved[1:]
        else:
            new = kk_morph.replace_stem(tok, form, approved)
        if new is None:
            notes.append(f"cannot rebuild endings of «{tok}»")
            return None, notes
        out = out[:start] + new + out[end:]
    return out, notes


def _matches(text: str, t: Term, exempt: set | None = None):
    ok = [m for f in t.ok_forms() for m in kk_find(text, f)]
    # a "forbidden" form that is the approved rendering of another term in the
    # same source is legitimate (source has both «кейс» and «сундук» -> «сандық» OK)
    forb = [f for f in t.forbidden if f.lstrip("=").lower() not in (exempt or set())]
    bad = [(s, e, tok, f) for f in forb for (s, e, tok) in kk_find(text, f)]
    # a forbidden hit inside an approved hit (stem overlap) is not a violation
    bad = [b for b in bad if not any(o[0] <= b[0] and b[1] <= o[1] for o in ok)]
    return ok, bad


def run(source: ArbFile, target: ArbFile, terms: list[Term], ref_en: ArbFile | None = None,
        cfg: GlossaryConfig | None = None) -> tuple[list[Finding], dict]:
    cfg = cfg or GlossaryConfig()
    stats = {"terms": len(terms), "keys_with_terms": 0, "forbidden": 0, "missing": 0}
    findings = []
    for key in source.keys():
        se, te = source.get(key), target.get(key)
        if se is None or te is None or not _eligible(se.value, te.value):
            continue
        en = ref_en.get(key).value if ref_en is not None and ref_en.get(key) else None
        problems, fixed, fix_ok, sev, conf = [], te.value, True, "low", 0.0
        present = [t for t in terms if t.in_source(se.value, en)]
        hit_any = bool(present)
        for t in present:
            exempt = {f.lower() for o in present if o is not t for f in o.ok_forms()}
            ok, bad = _matches(te.value, t, exempt)
            if bad and ok:
                continue
            if bad:
                kind = "forbidden"
                stats[kind] += 1
                toks = ", ".join(f"«{b[2]}»" for b in bad)
                problems.append({"term": t.ru, "kind": kind, "found": [b[2] for b in bad], "approved": t.kk,
                                 "text": (f"source has «{t.ru}» → glossary requires «{t.kk}»"
                                          + (f" (also OK: {', '.join(t.allowed)})" if t.allowed else "")
                                          + f"; target uses {toks}"
                                          + (f". Note: {t.notes}" if t.notes else ""))})
                if fix_ok:
                    new, notes = _fix(fixed, _matches(fixed, t, exempt)[1], t.kk)
                    if new is None:
                        fix_ok = False
                        problems[-1]["fix_note"] = "; ".join(notes)
                    else:
                        fixed = new
                sev = "high"
                conf = max(conf, 0.9)
            elif (not ok and t.strict and cfg.report_missing
                  and len(te.value.split()) <= cfg.missing_max_words):
                stats["missing"] += 1
                problems.append({"term": t.ru, "kind": "missing", "approved": t.kk,
                                 "text": f"source has «{t.ru}» but the target has no «{t.kk}»"
                                         + (f" or {', '.join(t.allowed)}" if t.allowed else "")
                                         + " (term dropped or replaced by an unlisted word)"})
                conf = max(conf, cfg.missing_confidence)
        if hit_any:
            stats["keys_with_terms"] += 1
        if not problems:
            continue
        suggestion = fixed if fix_ok and fixed != te.value else None
        if suggestion is not None and (not icu.is_valid(suggestion) or not icu.same_structure(te.value, suggestion)):
            suggestion = None
        findings.append(Finding(
            key=key, issue_type=ISSUE, severity=sev, source=se.value, target=te.value,
            evidence=" | ".join(p["text"] for p in problems), suggestion=suggestion,
            confidence=round(conf, 3), reviewed_by_human=False,
            details={"terms": problems}))
    rank = {"high": 2, "medium": 1, "low": 0}
    findings.sort(key=lambda f: (-rank[f.severity], -f.confidence, f.key))
    stats["reported"] = len(findings)
    return findings, stats
