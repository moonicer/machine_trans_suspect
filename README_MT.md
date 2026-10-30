# `machine_translation_suspect` — Kazakh MT / calque detector

Level-3 check of the Grand Mobile Qazaq Loc QA tool. Finds Kazakh strings that a
native player would recognise as machine-translated or copied from Russian:
calques, Russian grammar, non-Kazakh word order, officialese, English-MT style.

## Run

```bash
pip install -e ".[llm,dev]"          # anthropic is optional; core has no deps
pytest -q                             # 35 tests, no network

# offline, rules only (high precision, lower recall)
locqa check --source intl_ru.arb --target intl_kk.arb --glossary glossary.csv \
            --ref-en intl_en.arb --out report.json --html report_mt.html

# recommended: rules + LLM verification of candidates
export ANTHROPIC_API_KEY=...
locqa check ... --llm anthropic --llm-mode verify --pricing pricing.json --merge

# max recall: cheap model screens every multi-word string, strong model verifies
locqa check ... --llm anthropic --llm-mode scan \
            --screen-model claude-haiku-5-5 --verify-model claude-sonnet-5-5

# tune the threshold against the answer key / native labels (key,issue_type)
locqa eval --report report.json --labels labels.csv
```

`--merge` replaces only this issue type inside an existing `report.json`, so the
other 10 checks can write the same file. `report.meta.json` holds run stats,
token usage and USD cost (fill `pricing.json` from the official pricing page;
the tool never guesses prices).

## Integration into the main `locqa`

```python
from locqa.checks import machine_translation as mt
findings, stats = mt.run(src_arb, tgt_arb, judge, mt.MTConfig(), ref_en_arb, glossary_rows)
report.extend(f.to_dict() for f in findings)      # exact report.json schema + `details`
```

## Method

```
ARB → ICU-aware text extraction → eligibility → rule signals ─┬─► fusion → findings
                                                (p_rule)      │
                     candidates (p_rule ≥ 0.30) ─► LLM verify ┘
          (scan mode) other strings ≥3 words ─► LLM screen ─► verify
```

**0. Eligibility.** Empty, identical-to-source and mostly-Russian strings are
skipped: they belong to `untranslated`, `same_as_source`, `wrong_language`.
This prevents double-reporting one defect under two types (recall is scored
per issue type).

**1. Rule signals** (offline, each with the exact span and a reason):

| signal | weight | example (bad → good) |
|---|---|---|
| verb_before_object | 0.60 | «Сатып алу VIP PASS» → «VIP PASS сатып алу» (only Latin-script objects) |
| postposition_clause_initial | 0.70 | «Үшін гаражды…» → «Гаражды жалға алу үшін…» |
| possessive_agreement | 0.60 (сіздің) / 0.30 (біздің) | «Сіздің аккаунт» → «Сіздің аккаунтыңыз»; «біздің сайтта» is accepted colloquially |
| russian_function_word | 0.60 | «Көлік и үй» → «Көлік пен үй» |
| genitive_phrase_final (-ның/-нің) | 0.45 | «Деңгей ойыншының» → «Ойыншы деңгейі» (LLM candidate only) |
| reflexive_person_mismatch | 0.60 | «Өзінің прогресін сақтаңыз» → «Прогресіңізді сақтаңыз» |
| russian_inflection | 0.55 | «Технический үзіліс» → «Техникалық үзіліс» |
| genitive_phrase_final (-дың/-тің…) | 0.30 | same endings as informal past verbs (жеңдің) and 2nd-person possessives (құжатың) |
| calque_lexicon | 0.25–0.40 | «болып табылады», «жүзеге асыру», «тарапынан» |
| title_case | 0.35 | «Жаңа Ойын Режимі» (stronger evidence if EN is Title Case) |
| word_by_word_ratio | 0.20 | Kazakh has ≥ as many words as a preposition-heavy Russian source |

Combined with noisy-OR. Word lists exclude tokens that are also Kazakh words
(он «10», от «fire», у «poison», не «what», а/о/да), and «Қате орын алды» is
deliberately not treated as a calque — it is standard Kazakh UI. Placeholders and
ICU keywords are masked before analysis; every plural branch is checked.

**2. LLM judge.** One prompt per batch of 15, temperature 0, system prompt cached.
Each item carries key, description, RU source, EN reference, two neighbouring KK
strings (context for keys with no description), matching glossary rows, and the
rule hints labelled "may be wrong". The rubric names five categories and an
explicit do-not-flag list (loanwords, slang варн/мут/фура/ГК, typos, terms,
length — other checks own those) and says "when in doubt, ok".

Guards, applied to every answer:
- `evidence_span` must be an exact substring of the target, else the verdict is discarded (hallucination guard);
- a suggestion is dropped (finding kept) if it breaks ICU, changes the placeholder set,
  equals the original, or glues a suffix to a placeholder (`{name}-ға`), which would
  just create a `suffix_harmony` defect;
- answers are cached on disk by content hash → reruns are free and reproducible.

**3. Fusion.** `conf = 1 − (1 − p_rule)(1 − p_llm)` when the LLM confirms. An LLM
"ok" vetoes rule hits below 0.75; stronger rule hits survive at `0.6 × p_rule`
with an explicit note in the evidence. Offline mode reports only `p_rule ≥ 0.55`.
Severity: high = grammar broken / Russian words; medium = clearly translated;
low = stylistic.

## Human review

Every finding is emitted with `reviewed_by_human: false`; suggestions are never
applied automatically. The native speaker reviews the HTML report (evidence +
signals per row) and accepts / edits / rejects; only accepted or edited strings
go into `intl_kk_fixed.arb` with `reviewed_by_human: true`. Record the number of
reviewed strings in the main README methodology.

## Calibration on the handout (Oct 31)

First run on the real `intl_kk.arb` (5,479 source keys, 3,568 eligible for this check):

| step | offline findings | what was wrong |
|---|---|---|
| v1 rules | 30 | ~24 false positives |
| treat `\n` as a visual line break, not a clause end | — | genitive / postposition fired across UI line wraps |
| placeholder or quote before a postposition counts as its noun | — | «{price} үшін» flagged |
| skip fragments whose Russian ends in a preposition («Открыть за» + price) | — | concatenation, not MT |
| genitive endings shared with 2nd-person forms → LLM-only weight | — | «құжатың», «келдің» |
| «біздің X» without -мыз → weak | — | accepted colloquial usage |
| mid-sentence capitalised Russian-looking words = names (skins, NPCs) | — | «-ский» skin names |
| v2 rules | 6 | 0 false positives on manual inspection |

With a Claude judge over the rule candidates plus a scan of the most word-for-word strings
(`--llm replay` of the recorded verdicts), the run reports 15 findings, all with a
guarded suggestion. New rule from that review: `verb_before_object`.
Real strings are not reproduced here (data agreement); regression tests use paraphrases.

Data quirk for the whole team: `intl_kk.arb` repeats 109 string keys (10 with
different values). Dart's JSON decoding keeps the last value, so the last one is what ships.

## Calibration loop (repeat with native labels)

1. Run `--llm-mode scan` on `intl_kk.arb`; give the native speaker a random sample
   of 50 findings + 50 non-findings; label.
2. `locqa eval` → choose the threshold where precision ≥ 0.85 (judges review 100
   random findings), then maximise recall.
3. Add confirmed recurring calques from the real file to `CALQUES` / word lists,
   re-run the tests.

## Limitations

- Rules see surface forms only; no morphological analyser, so verb-final order and
  wrong case government are left to the LLM.
- The fixtures in `tests/fixtures` are synthetic; no competition data is in this repo
  (per the data agreement — keep it that way).
