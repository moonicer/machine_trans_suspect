# locqa — Localization QA for Flutter ARB (Grand Mobile · Qazaq Loc QA)

CLI that finds localization defects in a Flutter `intl_kk.arb` against its Russian source, explains each
finding, proposes a fix, and builds a corrected `intl_kk_fixed.arb` in which every changed string carries a
human-review flag.

**Scope of this repository.** Two of the 11 issue types are implemented and reviewed here:

| type | level | method |
|---|---|---|
| `machine_translation_suspect` | 3 – Hard | Kazakh rule signals + context-aware LLM judge, fused ([README_MT.md](README_MT.md)) |
| `glossary_violation` | 2 – Medium | glossary lookup with Russian inflection and Kazakh ending re-generation ([README_GLOSSARY.md](README_GLOSSARY.md)) |

The other nine types are not in this repo. The report schema, `--merge`, `locqa combine` and `locqa apply`
are type-agnostic, so their findings can be added to the same `report.json`. The 6 `orthography` entries in the
reviewed report were logged by the native reviewer while checking other strings; there is no orthography detector.

## Install and run

```bash
pip install -e ".[dev]"                # core has no dependencies; add ".[llm]" for Anthropic/OpenAI judges
pytest -q                              # 80 tests, no network

# 1. detect (offline, deterministic)
locqa check --source intl_ru.arb --target intl_kk.arb --glossary glossary.csv \
            --ref-en intl_en.arb --out report.json --html report.html

# 2. optional: LLM judge for machine_translation_suspect (recall; needs a key, see README_MT.md)
export ANTHROPIC_API_KEY=...            # or OPENAI_API_KEY (+ --llm openai --screen-model M --verify-model M)
locqa check ... --llm anthropic --llm-mode scan --pricing pricing.json --out report_llm.json

# 3. native speaker confirms / edits / rejects (see "Human review"), then merge the reports
locqa review  --report report_llm.json --decisions decisions.json --out report_llm.reviewed.json
locqa combine --reports mt.reviewed.json glossary.reviewed.json --out report.json     # + report.patch.json
locqa html    --report report.json --out report.html                                   # shows review status

# 4. corrected file
locqa apply --source intl_ru.arb --target intl_kk.arb --report report.json --out intl_kk_fixed.arb
```

Without a key, step 1 is fully offline: rule-based `machine_translation_suspect` (high precision, lower recall)
plus `glossary_violation`. The LLM raises recall; its output is **only ever applied after a human accepts it**.

## report.json

A list of findings with exactly the fields of the brief — `key, issue_type, severity, source, target, evidence,
suggestion, confidence, reviewed_by_human` — plus an optional `details` block (rule signals with spans and
weights, glossary terms, the LLM verdict and guard notes, the reviewer's note). `--no-details` drops it.
`evidence` is a sentence a reviewer can read, e.g. *Verb before object copies «Купить X»; Kazakh is object–verb*.
`report.html` shows the same rows with severity, confidence, evidence, signals and review status.

## Glossary

`glossary.csv` — required columns `ru, kk, en, notes`; optional `kk_allowed, kk_forbidden, en_sense, strict`
drive the check (details in README_GLOSSARY.md). The file is the only game-specific input: swap it to check another game.

## Human review and `intl_kk_fixed.arb`

* Every finding starts with `reviewed_by_human: false`. Nothing is applied automatically.
* `locqa review` records a decision per key (`accepted` / `edited` with the reviewer's own text / `rejected`
  drops the finding). Edits are re-validated for ICU syntax and placeholders; a broken edit aborts.
* `locqa apply` writes **only reviewed fixes** (pass `--include-unreviewed` to also apply the rest, flagged `false`).
  It edits the target **line by line**, so the repeated `"@@comment"` markers, duplicate keys and formatting survive
  (a JSON round trip would drop them). If a key is duplicated, every occurrence gets the fix (the last one wins in Dart).
* The flag is written next to each changed string, following the ARB `@key` convention:

  ```json
  "tesla_card_desc_2": "…fixed text…",
  "@tesla_card_desc_2": {"x-locqa-reviewed": true, "x-locqa-issue": "machine_translation_suspect",
                         "x-locqa-reviewer": "native Kazakh speaker (team)"},
  ```

  Flutter's `gen-l10n` reads resource attributes from the template locale; the extra `x-` attributes in a
  translation file are ignored. An existing `@key` object is merged, not replaced.
* `apply` refuses to write if any fix loses a placeholder of the source or breaks ICU syntax, and checks the output
  is valid JSON. It also writes `<out>.review.json` (before / after / issue / reviewer per key) as an audit log.
* Check in your Flutter project: `flutter gen-l10n` (the sandbox that produced the files had no Flutter SDK;
  JSON validity, placeholders and ICU were checked with the built-in parser and Node's `JSON.parse`).

### Methodology numbers (native-reviewed strings)

Result on the competition `intl_kk.arb` (5,479 source keys; the data itself is not in this repo):

* **170 findings, 161 distinct keys, all reviewed by a native Kazakh speaker** — 136 `machine_translation_suspect`,
  28 `glossary_violation`, 6 `orthography` (logged by the reviewer). **160 keys carry a text fix**; the 161st
  (`donate_open_case_for`) needs a client code change, so it has no text fix.
* Review decisions: MT first pass 18 (12 accepted, 6 edited); glossary 30 (19 accepted, 9 edited, 2 rejected);
  LLM verify pass: 2 new model findings (both edited) and 2 rule-only false alarms rejected; LLM scan: 113 groups
  of identical strings (107 accepted, 6 edited, 2 rejected).
* The LLM scan judged all 3,568 eligible strings and flagged 575. **About 430 lower-ranked findings (medium/low
  severity) were not reviewed**: they are in no patch, so they are not in `intl_kk_fixed.arb`. About 24 strings
  were never judged because 3 of 241 batches failed (fixed in the code, not re-run).
* The offline rules alone reproduce 10 of the 136 MT findings (all confirmed by the reviewer). The scan re-found 20 of
  the 23 strings the reviewer had fixed by hand before it ran. There is no answer key, so recall is **not** measured.

## Universality

* Works on any valid ARB: keys, `@key` metadata and `@@comment` markers are handled generically; ICU
  plural/select parsing is implemented in `locqa/icu.py` and every plural branch is checked.
* Nothing in the code is tied to this game: terms come from `glossary.csv`. The Kazakh rule signals
  (`locqa/checks/mt_rules.py`, `text_kk.py`, `kk_morph.py`) are language-specific by design; for a non-Kazakh
  target the MT check reports 0 findings instead of failing.
* Known weaknesses: the weak genitive rule (`-дың/-тың`) also matches 2nd-person verbs and possessives
  («келдің», «құжатың») and is the main source of false alarms; rules see surface forms only (no morphological
  analyser), so word order and case government rely on the LLM.

## Confidentiality

No competition data is in this repository: the fixtures in `tests/fixtures` are synthetic and the README examples
are paraphrased. Delete the local copies of the handout by the deadline set by the organizers. API keys are read
from the environment and never written to reports; the LLM cache (`.locqa_cache/`) holds competition strings and is git-ignored.
