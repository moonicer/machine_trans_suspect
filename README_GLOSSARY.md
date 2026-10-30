# `glossary_violation` and `glossary.csv`

## Run

```bash
locqa check --source intl_ru.arb --target intl_kk.arb --glossary glossary.csv \
            --ref-en intl_en.arb --checks glossary --out report.json --html report.html
locqa check ... --checks all        # glossary_violation + machine_translation_suspect
```

## glossary.csv

`ru, kk, en, notes` are the required deliverable columns; four optional columns
drive the check:

| column | meaning |
|---|---|
| `kk_allowed` | other acceptable renderings, `;`-separated (slang and standard: варн / ескерту) |
| `kk_forbidden` | renderings that violate the glossary (`тері` for скин). Prefix `=` for short words that must match as a whole word plus regular endings (`=іс`) |
| `en_sense` | regex on the English reference; disambiguates homographs (`пропуск` = pass, not skip) |
| `strict` | `yes`: the term must appear in short labels whenever the source has it |

### How the glossary was built
1. Mined the Russian source for the brief's terms (faction, family, Grand Coins,
   business, варн/мут/фура/ГК) and ~40 recurring game nouns; for each, counted the
   Kazakh renderings in the same keys.
2. The majority rendering became `kk`. Renderings that change the meaning became
   `kk_forbidden` (скин → тері "hide", инвентарь → түгендеу "stocktaking",
   ивент → оқиға "incident", кейс → іс "legal case"). Where the file is genuinely
   split between two acceptable forms (лидерборд / көшбасшылар кестесі), both are allowed.
3. Player slang the brief names (варн, мут, фура, ГК) is listed with the standard
   word as an allowed alternative, so moderation and chat strings can use either.
4. Кейс (loot box) and сундук (chest) are different items: «сандық» is forbidden for
   кейс and «кейс» for сундук.
5. Reviewed by the team's native Kazakh speaker.

## Detection

For every row whose Russian term occurs in the source (Russian inflection-aware:
«семью» matches семья, «семь дней» does not; «бан» does not match «банк»):

- **forbidden rendering, approved term absent** → `high`, confidence 0.9;
- forbidden rendering **next to** the approved term → not reported (common words
  such as «сыйлық» co-occur legitimately; `inconsistent_term` owns mixed usage);
- a forbidden form that is the approved rendering of **another** term in the same
  source is exempt (source has both кейс and сундук → «сандық» is fine);
- **strict term missing** from a label of ≤ 6 words → `low`, 0.55, no auto-fix.
  Long sentences paraphrase legitimately and are not checked for absence.

Empty, identical-to-source, Latin-script and mostly-Russian strings are skipped;
they belong to other issue types.

## Fixes: rebuilding Kazakh endings

Swapping a stem is not enough in Kazakh, so `locqa/kk_morph.py` parses the
ending chain of the wrong word (plural → possessive → case → -ғы) and regenerates
it for the approved stem with vowel harmony and consonant assimilation:

| wrong | parsed | fixed |
|---|---|---|
| терісін | тері + сі + н | скинін |
| түгендеуіңізден | түгендеу + іңіз + ден | инвентарыңыздан |
| оқиғасына | оқиға + сы + на | ивентіне |
| Іс-қимасындағы | іс-қима + сы + нда + ғы | Ивентіндегі |
| ойнаушыларға | ойнаушы + лар + ға | ойыншыларға |

If the ending is not a regular nominal chain (verb forms, derivation), the
suggestion is left empty rather than guessed. Case (UPPER / Title) is preserved.

## Result on the handout (Oct 31)

1,156 keys contain at least one glossary term → 30 findings: 26 forbidden
renderings (25 with an automatic fix) and 4 strict terms missing from short labels.
**Native review (Oct 31):** all 30 reviewed — 19 accepted, 9 edited, 2 rejected
(a term present only as a typo, which is an orthography issue; a sentence fragment whose
term sits in the other half). One accepted finding needs a code change: the client appends
the price after «Открыть кейс за», which Kazakh word order cannot follow.

While checking, the reviewer found 9 more problems in the same strings (garbled words,
wrong word for «скидки», wrong person/mood, capitalisation). They are logged under their
own issue types (`machine_translation_suspect`, `orthography`), because recall is scored
per type, and the final text of each key fixes all of its issues at once:

```bash
locqa combine --reports mt.reviewed.json glossary.reviewed.json manual.json --out report.json
# report.patch.json = one final string per key; fails if two reviewed fixes disagree
```

Combined result: 55 findings, all reviewed, 45 keys patched.
