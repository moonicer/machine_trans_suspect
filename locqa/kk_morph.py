"""Kazakh suffix analysis and regeneration (vowel harmony + consonant assimilation).

Purpose: when a glossary check replaces a wrong term with the approved one, the
case/number/possessive endings must be rebuilt for the new stem:
    терісін  (тері+сі+н)        -> скинін      (скин+і+н)
    түгендеуіңізден (…+іңіз+ден) -> инвентарыңыздан
    оқиғасына (оқиға+сы+на)      -> ивентіне
The same tables can drive suffix_harmony fixes.

Scope: the regular nominal chain  [plural] [possessive] [case]. Anything else
(verb forms, derivational suffixes) is reported as "cannot rebuild" so the
caller leaves the suggestion empty rather than inventing a word.
"""
from __future__ import annotations

BACK_V = set("аоұыяю")
FRONT_V = set("әеөүіэи")
VOWELS = BACK_V | FRONT_V | set("уё")
VOICELESS = set("кқпстфхцчшщ")
SONORANT_LRY = set("лрйу")
NASAL = set("мнң")
VOICED_ZJ = set("жз")
VOICED_OTHER = set("бвгд")

# morpheme type -> (back, front) allomorph lists, used for parsing
ALLOMORPHS = {
    "PL": (["лар", "дар", "тар"], ["лер", "дер", "тер"]),
    "P1": (["ым", "м"], ["ім", "м"]),
    "P2": (["ың", "ң"], ["ің", "ң"]),
    "P2P": (["ыңыз", "ңыз"], ["іңіз", "ңіз"]),
    "P1PL": (["ымыз", "мыз"], ["іміз", "міз"]),
    "P3": (["сы", "ы"], ["сі", "і"]),
    "GEN": (["ның", "дың", "тың"], ["нің", "дің", "тің"]),
    "DAT": (["ға", "қа", "на", "а"], ["ге", "ке", "не", "е"]),
    "ACC": (["ны", "ды", "ты", "н"], ["ні", "ді", "ті", "н"]),
    "LOC": (["да", "та", "нда"], ["де", "те", "нде"]),
    "ABL": (["дан", "тан", "нан"], ["ден", "тен", "нен"]),
    "INS": (["мен", "бен", "пен"], ["мен", "бен", "пен"]),
    "REL": (["ғы"], ["гі"]),          # -дағы/-дегі: locative + relative adjective
}
SLOTS = [["PL"], ["P1", "P2", "P2P", "P1PL", "P3"], ["GEN", "DAT", "ACC", "LOC", "ABL", "INS"], ["REL"]]


def _clean(stem: str) -> str:
    return stem.lower().rstrip("ь")


def is_front(word: str) -> bool:
    for ch in reversed(word.lower()):
        if ch in FRONT_V:
            return True
        if ch in BACK_V:
            return False
    return False


def _last(word: str) -> str:
    w = _clean(word)
    return w[-1] if w else ""


def _alternate(stem: str) -> str:
    """к->г, қ->ғ, п->б before a vowel-initial suffix (кейс is unaffected)."""
    m = {"к": "г", "қ": "ғ", "п": "б"}
    return stem[:-1] + m[stem[-1]] if stem and stem[-1] in m else stem


def generate(stem: str, morph: str, prev: str | None) -> str:
    """Return the suffix for `morph` attached after `stem` (already built).
    `prev` is the previous morpheme type (P3 changes the following case)."""
    f = 1 if is_front(stem) else 0
    last = _last(stem)
    vowel_end = last in VOWELS
    b = lambda back, front: front if f else back
    after_p3 = prev == "P3"
    after_p12 = prev in ("P1", "P2", "P2P", "P1PL")
    if morph == "PL":
        if vowel_end or last in "рйу":
            return b("лар", "лер")
        if last in "лмнңжз":
            return b("дар", "дер")
        return b("тар", "тер")
    if morph == "P3":
        return b("сы", "сі") if vowel_end else b("ы", "і")
    if morph == "P1":
        return "м" if vowel_end else b("ым", "ім")
    if morph == "P2":
        return "ң" if vowel_end else b("ың", "ің")
    if morph == "P2P":
        return b("ңыз", "ңіз") if vowel_end else b("ыңыз", "іңіз")
    if morph == "P1PL":
        return b("мыз", "міз") if vowel_end else b("ымыз", "іміз")
    if morph == "GEN":
        if vowel_end or last in NASAL:
            return b("ның", "нің")
        if last in VOICELESS or last in VOICED_OTHER:
            return b("тың", "тің")
        return b("дың", "дің")
    if morph == "DAT":
        if after_p3:
            return b("на", "не")
        if after_p12:
            return b("а", "е")
        return b("қа", "ке") if (last in VOICELESS or last in VOICED_OTHER) else b("ға", "ге")
    if morph == "ACC":
        if after_p3:
            return "н"
        if vowel_end:
            return b("ны", "ні")
        if last in VOICELESS or last in VOICED_OTHER:
            return b("ты", "ті")
        return b("ды", "ді")
    if morph == "LOC":
        if after_p3:
            return b("нда", "нде")
        return b("та", "те") if (last in VOICELESS or last in VOICED_OTHER) else b("да", "де")
    if morph == "ABL":
        if after_p3 or last in NASAL:
            return b("нан", "нен")
        return b("тан", "тен") if (last in VOICELESS or last in VOICED_OTHER) else b("дан", "ден")
    if morph == "REL":
        return b("ғы", "гі")       # always follows the vowel-final locative
    if morph == "INS":
        if vowel_end or last in NASAL or last in SONORANT_LRY:
            return "мен"
        if last in VOICED_ZJ:
            return "бен"
        return "пен"
    raise ValueError(morph)


def parse_suffix(suffix: str) -> list[str] | None:
    """Split a suffix string into morpheme types following slot order.
    Returns None if it is not a regular nominal chain."""
    s = suffix.lower()
    if not s:
        return []
    best = None

    def rec(pos: int, slot: int, acc: list[str]):
        nonlocal best
        if pos == len(s):
            if best is None or len(acc) < len(best):
                best = list(acc)
            return
        for si in range(slot, len(SLOTS)):
            for m in SLOTS[si]:
                if m == "REL" and (not acc or acc[-1] != "LOC"):
                    continue
                for form in set(ALLOMORPHS[m][0] + ALLOMORPHS[m][1]):
                    if s.startswith(form, pos):
                        acc.append(m)
                        rec(pos + len(form), si + 1, acc)
                        acc.pop()
    rec(0, 0, [])
    return best


def stem_variants(stem: str) -> list[str]:
    """Surface forms a stem can take inside a word (final alternation, ь)."""
    s = _clean(stem)
    v = {s, stem.lower(), _alternate(s)}
    return sorted(v, key=len, reverse=True)


def split_token(token: str, stem: str) -> tuple[str, str] | None:
    """If token = stem(+alternation) + regular suffix, return (stem_part, suffix)."""
    t = token.lower()
    for sv in stem_variants(stem):
        if t.startswith(sv):
            rest = t[len(sv):]
            if parse_suffix(rest) is not None:
                return token[:len(sv)], token[len(sv):]
    return None


def inflect(stem: str, morphs: list[str]) -> str:
    base = _clean(stem) if morphs else stem
    out = base
    prev = None
    for m in morphs:
        suf = generate(out, m, prev)
        if suf and suf[0] in VOWELS:
            out = _alternate(out)
        out += suf
        prev = m
    return out


def _match_case(src: str, dst: str) -> str:
    if src.isupper():
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


def replace_stem(token: str, old_stem: str, new_stem: str) -> str | None:
    """терісін, тері, скин -> скинін. None if the token is not old_stem + a
    regular nominal suffix chain."""
    sp = split_token(token, old_stem)
    if sp is None:
        return None
    morphs = parse_suffix(sp[1])
    if morphs is None:
        return None
    return _match_case(token, inflect(new_stem, morphs))
