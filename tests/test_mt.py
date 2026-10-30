import json
from pathlib import Path

import pytest

from locqa import icu
from locqa.arb import ArbFile
from locqa.checks import machine_translation as mt
from locqa.checks.mt_llm import JudgeItem, StaticJudge, Usage, apply_guards
from locqa.checks.mt_rules import eligible, rule_signals
from locqa.cli import main

FX = Path(__file__).parent / "fixtures"


def names(src, tgt, en=None):
    return {s.name for s in rule_signals(src, tgt, en)}


# ---------------- ICU ----------------
def test_icu_placeholders_and_plural():
    s = "{count, plural, =0{Нет монет} one{{count} монета} other{{count} монет у {name}}}"
    assert icu.placeholders(s) == {"count", "name"}
    segs = icu.text_segments(s)
    assert any("монет" in x for x in segs) and all("plural" not in x for x in segs)


@pytest.mark.parametrize("bad", ["{name", "text }", "{n, plural, one{x}}"])
def test_icu_invalid(bad):
    assert not icu.is_valid(bad)


def test_icu_quotes():
    assert icu.placeholders("It''s '{literal}' {x}") == {"x"}


# ---------------- rules: positives ----------------
def test_postposition_first():
    assert "postposition_clause_initial" in names("Для аренды нажмите", "Үшін жалға алу басыңыз")


def test_genitive_final():
    assert "genitive_phrase_final" in names("Уровень игрока", "Деңгей ойыншының")


def test_possessive_agreement():
    assert "possessive_agreement" in names("Ваш профиль заморожен", "Сіздің профиль бұғатталды")


def test_reflexive_person():
    assert "reflexive_person_mismatch" in names("Сохраните свой прогресс", "Өзінің прогресін сақтаңыз")


def test_russian_function_word():
    assert "russian_function_word" in names("Купить машину и дом", "Көлік и үй сатып алу")


def test_russian_inflection():
    assert "russian_inflection" in names("Технический перерыв", "Технический үзіліс басталды")


def test_calque():
    assert "calque_lexicon" in names("Фракция является организацией", "Фракция ұйым болып табылады")


def test_title_case_with_en_ref():
    sigs = rule_signals("Новый игровой режим", "Жаңа Ойын Режимі", "New Game Mode")
    assert any(s.name == "title_case" and "English" in s.reason for s in sigs)


# ---------------- rules: negatives (precision) ----------------
@pytest.mark.parametrize("src,tgt", [
    ("Произошла ошибка", "Қате орын алды"),            # standard UI phrase, not a calque
    ("Ты победил!", "Сен жеңдің!"),                    # 2nd-person past verb, not genitive
    ("Вы получили варн", "Сіз варн алдыңыз"),            # player slang
    ("Игрок {name} вошёл в игру", "Ойыншы {name} ойынға кірді"),
    ("Ваш профиль заморожен", "Сіздің профиліңіз бұғатталды"),
    ("Уровень игрока", "Ойыншы деңгейі"),
    ("Он получил 10 монет", "Ол он монета алды"),        # «он» = ten in Kazakh
    ("Гараж арендован", "Гараж жалға алынды"),
    ("Сохраните свой прогресс", "Прогресіңізді сақтаңыз"),
    ("Бесплатно", "Ақысыз алыңыз"),                    # -сыз 'without' is not 2nd person
    # regressions found on the real file (paraphrased, no competition data):
    ("Продать за {amount}", "{amount} үшін сату"),         # placeholder precedes postposition
    ("Приз за миссии", "АПТАЛЫҚ МИССИЯЛАР\nҮШІН ЖҮЛДЕ"),  # newline = visual wrap
    ("Узнавайте о новостях", "Жаңалықтар\nтуралы біліп отырыңыз"),
    ("Пополняется через «+»", "Отын («+» арқылы толтырылады)"),
    ("Это твой документ", "Бұл сенің басты құжатың"),        # 2nd-person possessive -ың
    ("Посетите наш форум", "Біздің форумға кіріңіз"),         # colloquially accepted
    ("Скины: Петровский", "Скиндер: Петровский және Ноклип"),  # proper name, not Russian grammar
])
def test_clean_strings_have_no_strong_signal(src, tgt):
    assert all(s.weight < 0.4 for s in rule_signals(src, tgt)), rule_signals(src, tgt)


def test_verb_before_latin_object():
    assert "verb_before_object" in names("Купить VIP PASS", "Сатып алу VIP PASS")
    assert "verb_before_object" not in names("VIP PASS купить", "VIP PASS сатып алу")
    assert "verb_before_object" not in names("Цена аренды", "Жалға алу бағасы")


def test_fragment_source_skips_word_order_rule():
    # «Открыть за» + price appended by code: concatenation issue, not MT
    assert "postposition_clause_initial" not in names("Открыть за", "үшін ашу")
    assert "postposition_clause_initial" in names("Открыть за деньги", "үшін ақшаға ашу")


def test_eligibility_routes_other_issue_types():
    assert not eligible("Привет", "")[0]
    assert not eligible("Купить дом", "Купить дом")[0]
    assert not eligible("Добро пожаловать", "Добро пожаловать в игру мы рады вас видеть")[0]
    assert eligible("Купить дом", "Үй сатып алу")[0]


# ---------------- LLM guards ----------------
def _it(target="Деңгей ойыншының", source="Уровень игрока"):
    return JudgeItem(id=0, key="k", source=source, target=target)


def test_guard_discards_ungrounded_span():
    raw = {"verdict": "mt", "confidence": 0.9, "evidence_span": "жоқ сөз", "suggestion": "Ойыншы деңгейі"}
    assert apply_guards(_it(), raw, "m") is None


def test_guard_drops_placeholder_breaking_suggestion():
    it = _it("Ойыншы {name} кірді ойынға", "Игрок {name} вошёл в игру")
    raw = {"verdict": "mt", "confidence": 0.8, "evidence_span": "кірді ойынға",
           "suggestion": "Ойыншы ойынға кірді"}
    v = apply_guards(it, raw, "m")
    assert v and v.suggestion is None and "placeholder" in v.guard_notes[0]


def test_guard_drops_glued_suffix():
    it = _it("Сыйлық жіберілді {name} үшін", "Подарок отправлен {name}")
    raw = {"verdict": "mt", "confidence": 0.8, "evidence_span": "{name} үшін",
           "suggestion": "Сыйлық {name}-ға жіберілді"}
    assert apply_guards(it, raw, "m").suggestion is None


def test_guard_keeps_good_suggestion():
    raw = {"verdict": "mt", "confidence": 0.85, "evidence_span": "Деңгей ойыншының",
           "suggestion": "Ойыншы деңгейі", "severity": "medium"}
    assert apply_guards(_it(), raw, "m").suggestion == "Ойыншы деңгейі"


# ---------------- orchestrator ----------------
def _files():
    return ArbFile(FX / "intl_ru.arb"), ArbFile(FX / "intl_kk.arb"), ArbFile(FX / "intl_en.arb")


EXPECTED_MT = {"garage_rent_hint", "account_blocked", "buy_car_house", "save_progress",
               "player_level_title", "faction_about", "new_mode_title"}
CLEAN = {"shop_buy_button", "error_generic", "coins_count", "player_joined", "you_won",
         "warn_received", "garage_rent_button", "welcome_ru"}


def test_rules_only_is_high_precision():
    ru, kk, en = _files()
    f, _ = mt.run(ru, kk, None, mt.MTConfig(llm_mode="none"), en)
    keys = {x.key for x in f}
    assert keys <= EXPECTED_MT and not keys & CLEAN
    assert {"garage_rent_hint", "account_blocked", "buy_car_house"} <= keys
    # ambiguous genitive alone is an LLM candidate, not an offline finding
    assert "player_level_title" not in keys


def test_llm_fusion_recall_and_veto():
    ru, kk, en = _files()
    answers = {
        "faction_about": {"verdict": "mt", "confidence": 0.8, "evidence_span": "болып табылады",
                          "suggestion": "Фракция — ойыншылар ұйымы", "categories": ["calque"],
                          "explanation": "Calque of «является»."},
        "new_mode_title": {"verdict": "mt", "confidence": 0.7, "evidence_span": "Жаңа Ойын Режимі",
                           "suggestion": "Жаңа ойын режимі", "categories": ["english_mt"],
                           "explanation": "English-style Title Case."},
        "save_progress": {"verdict": "ok", "confidence": 0.6},   # LLM vetoes a 0.6 rule hit
    }
    f, stats = mt.run(ru, kk, StaticJudge(answers), mt.MTConfig(llm_mode="verify"), en)
    by = {x.key: x for x in f}
    assert "faction_about" in by and by["faction_about"].suggestion == "Фракция — ойыншылар ұйымы"
    assert "new_mode_title" in by
    assert "save_progress" not in by
    assert not set(by) & CLEAN
    assert stats["verified"] >= 5
    for x in f:
        d = x.to_dict()
        assert set(d) >= {"key", "issue_type", "severity", "source", "target", "evidence",
                          "suggestion", "confidence", "reviewed_by_human"}
        assert d["reviewed_by_human"] is False and x.evidence


def test_strong_rule_survives_llm_ok_with_note():
    ru, kk, en = _files()
    answers = {"garage_rent_hint": {"verdict": "ok", "confidence": 0.5},
               "buy_car_house": {"verdict": "ok", "confidence": 0.5}}
    cfg = mt.MTConfig(llm_mode="verify", strong_rule_threshold=0.6)
    f, _ = mt.run(ru, kk, StaticJudge(answers), cfg, en)
    g = next(x for x in f if x.key == "garage_rent_hint")
    assert "LLM reviewer judged" in g.evidence and g.confidence < 0.6


def test_usage_cost():
    class U: input_tokens = 1_000_000; output_tokens = 500_000; cache_read_input_tokens = 0; cache_creation_input_tokens = 0
    u = Usage({"m": {"input": 1.0, "output": 2.0}})
    u.add("m", U)
    assert u.report()["usd_total"] == 2.0
    assert Usage().report()["usd_total"] is None


# ---------------- CLI ----------------
def test_cli_end_to_end(tmp_path):
    out, html = tmp_path / "report.json", tmp_path / "report.html"
    rc = main(["check", "--source", str(FX / "intl_ru.arb"), "--target", str(FX / "intl_kk.arb"),
               "--glossary", str(FX / "glossary.csv"), "--ref-en", str(FX / "intl_en.arb"),
               "--out", str(out), "--html", str(html)])
    assert rc == 0
    data = json.loads(out.read_text("utf-8"))
    assert data and all(d["issue_type"] == "machine_translation_suspect" for d in data)
    assert "machine_translation_suspect" in html.read_text("utf-8")


def test_cli_merge_keeps_other_issue_types(tmp_path):
    out = tmp_path / "report.json"
    out.write_text(json.dumps([{"key": "x", "issue_type": "untranslated"}]), "utf-8")
    main(["check", "--source", str(FX / "intl_ru.arb"), "--target", str(FX / "intl_kk.arb"),
          "--out", str(out), "--merge"])
    types = {d["issue_type"] for d in json.loads(out.read_text("utf-8"))}
    assert types == {"untranslated", "machine_translation_suspect"}


# ---------------- Anthropic judge with a fake client (no network) ----------------
def test_anthropic_judge_batches_caches_and_counts(tmp_path, monkeypatch):
    import sys, types
    calls = []

    class Msgs:
        def create(self, **kw):
            calls.append(kw)
            items = json.loads(kw["messages"][0]["content"].split("\n", 1)[1])
            arr = [{"id": it["id"], "verdict": "mt" if "ойыншының" in it["kk"] else "ok",
                    "confidence": 0.9, "evidence_span": "Деңгей ойыншының",
                    "suggestion": "Ойыншы деңгейі", "severity": "medium",
                    "categories": ["word_order"], "explanation": "Possessor after head."} for it in items]
            usage = types.SimpleNamespace(input_tokens=100, output_tokens=50,
                                          cache_read_input_tokens=0, cache_creation_input_tokens=0)
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="```json\n" + json.dumps(arr, ensure_ascii=False) + "\n```")],
                                         usage=usage)

    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda: types.SimpleNamespace(messages=Msgs())
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    from locqa.checks.mt_llm import AnthropicJudge
    usage = Usage()
    j = AnthropicJudge(cache_dir=tmp_path, batch_size=2, usage=usage)
    ru, kk, en = _files()
    f1, _ = mt.run(ru, kk, j, mt.MTConfig(llm_mode="scan", screen_model="m", verify_model="m"), en)
    n_calls = len(calls)
    assert n_calls >= 2 and calls[0]["temperature"] == 0
    assert calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert any(x.key == "player_level_title" and x.suggestion == "Ойыншы деңгейі" for x in f1)
    f2, _ = mt.run(ru, kk, j, mt.MTConfig(llm_mode="scan", screen_model="m", verify_model="m"), en)
    assert len(calls) == n_calls and usage.cache_hits > 0          # second run fully cached
    assert [x.key for x in f1] == [x.key for x in f2]
