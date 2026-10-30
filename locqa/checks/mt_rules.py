"""Stage 1 of machine_translation_suspect: explainable, offline rule signals.

Every signal carries the exact target span that triggered it, a weight in
(0, 1) read as "probability this alone indicates MT", and a human-readable
reason. Signals are combined with a noisy-OR in the orchestrator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .. import icu
from ..text_kk import (CALQUES, RU_FUNCTION_WORDS, GENITIVE, PERSONAL_GENITIVES, POSTPOSITIONS, PH,
                       clauses, has_kk_letters, is_russian_token, is_title_case,
                       language_profile, words)

RU_PREPS = {"в", "во", "на", "для", "с", "со", "по", "к", "ко", "из", "от", "до",
            "за", "при", "без", "через", "о", "об", "у", "под", "над", "после"}


@dataclass
class Signal:
    name: str
    weight: float
    span: str
    reason: str
    severity: str = "medium"

    def to_dict(self):
        return {"name": self.name, "weight": self.weight, "span": self.span,
                "reason": self.reason, "severity": self.severity}


def eligible(source: str, target: str) -> tuple[bool, str]:
    """MT-suspect only applies to text that is mostly Kazakh already.
    Fully Russian / identical / empty strings belong to other checks."""
    if not target.strip():
        return False, "empty (untranslated)"
    if target.strip() == source.strip():
        return False, "identical to source (same_as_source)"
    prof = language_profile(" ".join(icu.text_segments(target)))
    if prof["n_cyr"] < 2:
        return False, "too short for a fluency judgement"
    if prof["ru_share"] > 0.5 or (prof["kk_share"] == 0 and prof["n_cyr"] >= 4):
        return False, "predominantly Russian (wrong_language)"
    return True, ""


def _russian_tokens(seg: str) -> list[Signal]:
    out = []
    for w in words(seg):
        if is_russian_token(w):
            fw = w.lower() in RU_FUNCTION_WORDS
            out.append(Signal(
                "russian_function_word" if fw else "russian_inflection",
                0.6 if fw else 0.55, w,
                (f"Russian {'function word' if fw else 'inflection'} «{w}» inside Kazakh text "
                 "(Russian grammar carried over)"),
                "high"))
    return out


def _postposition_first(seg: str) -> list[Signal]:
    out = []
    for cl in clauses(seg):
        ws = [w for w in words(cl) if w != PH]
        if ws and ws[0].lower() in POSTPOSITIONS and len(ws) >= 2:
            out.append(Signal("postposition_clause_initial", 0.7, ws[0],
                              f"«{ws[0]}» is a postposition but opens the clause — "
                              "a Russian/English preposition was copied in place (word order)", "high"))
    return out


def _genitive_phrase_final(seg: str) -> list[Signal]:
    """Kazakh genitive modifiers precede their head (ойыншының деңгейі).
    A genitive at the end of a phrase mirrors Russian «уровень игрока»."""
    out = []
    for cl in clauses(seg):
        ws = [w for w in words(cl)]
        if len(ws) < 2:
            continue
        last, prev = ws[-1], ws[-2].lower()
        if last == PH or last.lower() in PERSONAL_GENITIVES or prev in {"сен", "сендер"}:
            continue  # «Сен жеңдің» = 2nd-person past verb, not a genitive
        if GENITIVE.search(last.lower()) and len(last) >= 5:
                # -ның/-нің is unambiguous; -дың/-дің/-тың/-тің also ends informal
                # 2nd-person past verbs (алдың, жеңдің), so it weighs less.
                w = 0.6 if last.lower().endswith(("ның", "нің")) else 0.4
                out.append(Signal("genitive_phrase_final", w, f"{ws[-2]} {last}",
                                  f"genitive «{last}» ends the phrase; Kazakh puts the possessor "
                                  "before the possessed noun (Russian noun–genitive order copied)", "medium"))
    return out


def _possessive_agreement(seg: str) -> list[Signal]:
    """«Сіздің аккаунт» copies Russian «ваш аккаунт»; Kazakh needs the
    matching possessive suffix on the head: «Сіздің аккаунтыңыз»."""
    out = []
    ws = words(seg)
    req = {"сіздің": ("ңыз", "ңіз"), "біздің": ("мыз", "міз")}
    for i, w in enumerate(ws):
        need = req.get(w.lower())
        if not need:
            continue
        window = ws[i + 1:i + 4]
        if not window or any(t == PH for t in window[:1]):
            continue
        if not any(s in t.lower() for t in window for s in need):
            out.append(Signal("possessive_agreement", 0.6, " ".join([w] + window[:2]),
                              f"«{w}» is not followed by a noun with the matching possessive suffix "
                              f"(-{need[0]}/-{need[1]}); this copies the Russian possessive pronoun", "high"))
    return out


def _reflexive_person(seg: str) -> list[Signal]:
    low = seg.lower()
    second = re.search(r"\bсіз|ңыз|ңіз", low)
    m = re.search(r"\bөзін(ің|е|ен|де)?\b", low)
    if second and m:
        return [Signal("reflexive_person_mismatch", 0.6, m.group(0),
                       f"«{m.group(0)}» (3rd person) used in a 2nd-person (Сіз) string — "
                       "literal rendering of Russian «свой»; use «өз ...ңыз» / «өзіңіз»", "medium")]
    return []


def _calques(seg: str) -> list[Signal]:
    out = []
    for rx, w, ru, note in CALQUES:
        m = rx.search(seg)
        if m:
            out.append(Signal("calque_lexicon", w, m.group(0), f"{note} (← {ru})", "low"))
    return out


def _title_case(source: str, target: str, ref_en: str | None) -> list[Signal]:
    if is_title_case(target) and not is_title_case(source):
        via = " — matches the English reference, suggesting MT from English" if ref_en and is_title_case(ref_en) else ""
        return [Signal("title_case", 0.35, target[:60],
                       "Every word is capitalised; Kazakh (like Russian) capitalises only the first word" + via,
                       "low")]
    return []


def _preposition_mirror(source: str, target: str) -> list[Signal]:
    """Kazakh is agglutinative: Russian prepositions become suffixes, so a good
    translation of a preposition-heavy sentence has FEWER words. Equal or more
    words suggests word-by-word transfer. Weak signal; never fires alone."""
    sw = [w.lower() for w in words(" ".join(icu.text_segments(source))) if w != PH]
    tw = [w for w in words(" ".join(icu.text_segments(target))) if w != PH]
    n_prep = sum(1 for w in sw if w in RU_PREPS)
    if len(sw) >= 6 and n_prep >= 2 and len(tw) >= len(sw):
        return [Signal("word_by_word_ratio", 0.2, "",
                       f"{len(tw)} Kazakh words for {len(sw)} Russian words with {n_prep} prepositions; "
                       "natural Kazakh is usually shorter in words", "low")]
    return []


def rule_signals(source: str, target: str, ref_en: str | None = None) -> list[Signal]:
    sigs: list[Signal] = []
    for seg in icu.text_segments(target):
        sigs += _russian_tokens(seg)
        sigs += _postposition_first(seg)
        sigs += _genitive_phrase_final(seg)
        sigs += _possessive_agreement(seg)
        sigs += _reflexive_person(seg)
        sigs += _calques(seg)
    sigs += _title_case(source, target, ref_en)
    sigs += _preposition_mirror(source, target)
    # dedupe identical (name, span) across plural branches
    seen, out = set(), []
    for s in sigs:
        k = (s.name, s.span.lower())
        if k not in seen:
            seen.add(k); out.append(s)
    return out


def noisy_or(weights) -> float:
    p = 1.0
    for w in weights:
        p *= (1.0 - w)
    return 1.0 - p
