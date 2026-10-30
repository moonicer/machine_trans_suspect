"""Kazakh / Russian surface-text utilities.

Everything here is deliberately conservative: word lists exclude tokens that
are also valid Kazakh words (он 'ten', от 'fire', у 'poison', не 'what',
о / а / да interjections/particles), because the MT check is scored on
precision by human reviewers.
"""
from __future__ import annotations

import re

from .icu import PH

KK_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
WORD_RE = re.compile(r"⟨X⟩|[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]+(?:[-'][A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]+)*")
# "\n" is NOT a boundary: in UI strings it is a visual line break inside one phrase.
CLAUSE_BREAK = re.compile(r"[.!?…:;,—–]")

# Russian function words that do not exist as standalone Kazakh words.
RU_FUNCTION_WORDS = {
    "и", "в", "во", "для", "или", "что", "чтобы", "который", "которая", "которое",
    "которые", "это", "этот", "эта", "эти", "как", "если", "уже", "ещё", "еще",
    "можно", "нельзя", "нужно", "также", "тоже", "только", "очень", "был", "была",
    "были", "будет", "есть", "при", "без", "через", "после", "перед", "между",
    "из", "до", "по", "на", "за", "к", "ко", "об", "с", "со", "под", "над", "но",
    "все", "всё", "его", "её", "их", "нет", "ваш", "ваша", "ваше", "ваши",
    "вашего", "свой", "своё", "свою", "свои", "мы", "вы", "она", "они", "чем",
}

# Russian inflectional endings that native Kazakh words never carry and that
# assimilated loanwords lose in Kazakh (технический -> техникалық).
RU_ENDINGS = ("ться", "тся", "ый", "ые", "ого", "ому", "ыми", "ных", "ного",
              "ному", "ешь", "ишь", "ают", "яют", "ует", "ает",
              "ский", "ская", "ское", "ские", "цкий", "ная", "ное")

GENITIVE = re.compile(r"(ның|нің|дың|дің|тың|тің)$")
PERSONAL_GENITIVES = {"менің", "сенің", "оның", "біздің", "сіздің", "сендердің",
                      "сіздердің", "олардың", "кімнің", "мұның", "соның", "бұның"}

# Postpositions: in Kazakh they FOLLOW their noun; at the start of a clause they
# betray a Russian/English preposition copied in place (Для... -> Үшін...).
POSTPOSITIONS = {"үшін", "туралы", "арқылы", "бойынша", "сияқты", "жайлы", "дейін"}

# Bureaucratic calques (weight, Russian origin, note). Kept short on purpose:
# each was chosen because it is unnatural in casual game UI, not because it is
# ungrammatical. "Қате орын алды" is intentionally NOT here: it is standard UI.
CALQUES = [
    (re.compile(r"болып\s+табыла", re.I), 0.40, "является",
     "«болып табылады» is a calque of «является»; game UI normally uses a plain copula or rephrases"),
    (re.compile(r"жүзеге\s+асыр", re.I), 0.25, "осуществлять",
     "«жүзеге асыру» is officialese for «осуществить»; a concrete verb (жасау, орындау) reads naturally"),
    (re.compile(r"\bтарапынан\b", re.I), 0.25, "со стороны",
     "«...тарапынан» mirrors the Russian passive «со стороны»; Kazakh prefers an active subject"),
    (re.compile(r"\bөз\s+кезегінде\b", re.I), 0.30, "в свою очередь",
     "«өз кезегінде» is a word-for-word copy of «в свою очередь»"),
    (re.compile(r"\bнегізгі\s+болып\b", re.I), 0.30, "основным является",
     "copy of the Russian «основным является» construction"),
]


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def is_cyrillic_word(w: str) -> bool:
    return bool(re.fullmatch(r"[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі-]+", w))


def has_kk_letters(w: str) -> bool:
    return any(c in KK_LETTERS for c in w)


def is_russian_token(w: str) -> bool:
    lw = w.lower()
    if w == PH or has_kk_letters(w) or not is_cyrillic_word(w):
        return False
    if lw in RU_FUNCTION_WORDS:
        return True
    return len(lw) >= 5 and lw.endswith(RU_ENDINGS)


def language_profile(text: str) -> dict:
    ws = [w for w in words(text) if w != PH]
    cyr = [w for w in ws if is_cyrillic_word(w)]
    kk = [w for w in cyr if has_kk_letters(w)]
    ru = [w for w in cyr if is_russian_token(w)]
    n = max(len(cyr), 1)
    return {"n_words": len(ws), "n_cyr": len(cyr), "kk_share": len(kk) / n,
            "ru_share": len(ru) / n, "latin": len(ws) - len(cyr)}


def clauses(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    return [c.strip() for c in CLAUSE_BREAK.split(text) if c.strip()]


def clause_initial_word(clause: str) -> str | None:
    """First token of a clause, only if nothing (word, placeholder, quote,
    symbol or markup) precedes it."""
    m = WORD_RE.search(clause)
    if not m or m.group(0) == PH:
        return None
    prefix = re.sub(r"<[^>]*>", "", clause[:m.start()]).strip()
    return m.group(0) if not prefix else None


def is_title_case(text: str) -> bool:
    ws = [w for w in words(text) if w != PH and len(w) >= 2 and w.isalpha()]
    if len(ws) < 3 or all(w.isupper() for w in ws):
        return False
    caps = sum(1 for w in ws[1:] if w[0].isupper() and not w.isupper())
    return caps / (len(ws) - 1) >= 0.8
