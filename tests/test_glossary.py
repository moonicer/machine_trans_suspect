import json
from pathlib import Path

import pytest

from locqa import kk_morph
from locqa.arb import ArbFile
from locqa.checks import glossary_violation as gv
from locqa.cli import main
from locqa.glossary import Term, kk_find, load_terms

FX = Path(__file__).parent / "fixtures"
TERMS = load_terms(FX / "glossary_terms.csv")


@pytest.mark.parametrize("token,old,new,want", [
    ("терісін", "тері", "скин", "скинін"),
    ("Терілер", "тері", "скин", "Скиндер"),
    ("түгендеуіңізден", "түгендеу", "инвентарь", "инвентарыңыздан"),
    ("оқиғасына", "оқиға", "ивент", "ивентіне"),
    ("оқиғаны", "оқиға", "ивент", "ивентті"),
    ("ойнаушыларға", "ойнаушы", "ойыншы", "ойыншыларға"),
    ("есепті", "есеп", "аккаунт", "аккаунтты"),
    ("ЕСЕПТІ", "есеп", "аккаунт", "АККАУНТТЫ"),
    ("сандығы", "сандық", "кейс", "кейсі"),
    ("кейсті", "кейс", "сандық", "сандықты"),
    ("көліктің", "көлік", "скин", "скиннің"),
    ("Іс-қимасындағы", "іс-қима", "ивент", "Ивентіндегі"),
    ("ойынға", "ойын", "ивент", "ивентке"),
    ("үйге", "үй", "пәтер", "пәтерге"),
])
def test_replace_stem_rebuilds_endings(token, old, new, want):
    assert kk_morph.replace_stem(token, old, new) == want


def test_replace_stem_refuses_non_nominal():
    assert kk_morph.replace_stem("терілеуге", "тері", "скин") is None   # verb/derivation: no guess


def test_ru_homographs():
    fam = Term(ru="семья", kk="отбасы")
    assert fam.in_source("Пригласить в семью") and not fam.in_source("семь дней подряд")
    ban = Term(ru="бан", kk="бан")
    assert ban.in_source("Выдан бан на 3 дня") and not ban.in_source("Пополнить банк")
    pas = Term(ru="пропуск", kk="пасс", en_sense=r"\bpass\b")
    assert pas.in_source("Купить пропуск", "Buy pass") and not pas.in_source("Пропуск уровня", "Skip level")


def test_short_exact_form_does_not_match_inside_words():
    assert kk_find("Іс ашылуда", "=іс")
    assert not kk_find("Бұл істеп тұр", "=іс")     # «істеп» is a verb, not «іс» + case


def _arb(tmp, name, d):
    p = tmp / name
    p.write_text(json.dumps(d, ensure_ascii=False), "utf-8")
    return ArbFile(p)


def _run(tmp, pairs, en=None):
    ru = _arb(tmp, "ru.arb", {k: v[0] for k, v in pairs.items()})
    kk = _arb(tmp, "kk.arb", {k: v[1] for k, v in pairs.items()})
    ref = _arb(tmp, "en.arb", en) if en else None
    f, _ = gv.run(ru, kk, TERMS, ref)
    return {x.key: x for x in f}


def test_forbidden_term_flagged_with_rebuilt_fix(tmp_path):
    by = _run(tmp_path, {
        "a": ("Выберите скин машины", "Көліктің терісін таңдаңыз"),
        "b": ("Скины", "Терілер"),
        "c": ("Удалено из инвентаря навсегда", "Түгендеуіңізден біржола өшірілді"),
        "d": ("Список игроков", "Ойнаушылар тізімі"),
    })
    assert by["a"].suggestion == "Көліктің скинін таңдаңыз" and by["a"].severity == "high"
    assert by["b"].suggestion == "Скиндер"
    assert by["c"].suggestion == "Инвентарыңыздан біржола өшірілді"
    assert by["d"].suggestion == "Ойыншылар тізімі"
    assert "тері" in by["a"].evidence and "скин" in by["a"].evidence


def test_cross_term_exemption_and_approved_present(tmp_path):
    by = _run(tmp_path, {
        # source names both «кейс» and «сундук»: «сандық» is the chest, not a violation
        "both": ("Открой кейс и забери сундук", "Кейсті аш та сандықты ал"),
        # approved term already present -> not this check's business
        "ok": ("Новые скины", "Жаңа скиндер"),
        # forbidden word used where the source has no glossary term
        "free": ("Кожаная куртка", "Тері күрте"),
    })
    assert by == {}


def test_missing_strict_term_only_in_short_labels(tmp_path):
    by = _run(tmp_path, {
        "label": ("Редкие кейсы", "Сирек"),
        "long": ("Открывай кейсы каждый день и получай редкие награды за активность в игре",
                 "Күн сайын ашып, ойындағы белсенділік үшін сирек сыйлықтар ал"),
    })
    assert set(by) == {"label"} and by["label"].severity == "low" and by["label"].suggestion is None


def test_skips_other_issue_types(tmp_path):
    by = _run(tmp_path, {
        "same": ("Скины", "Скины"),               # same_as_source
        "empty": ("Скины", ""),                   # untranslated
        "uz": ("Новые скины", "Yangi terilar"),   # Latin: wrong_language
    })
    assert by == {}


def test_en_sense_filters_skip_meaning(tmp_path):
    by = _run(tmp_path, {"p": ("Пропуск уровня", "Деңгейді өткізу"), "q": ("Купить пропуск", "Өткізу сатып алу")},
              en={"p": "Skip level", "q": "Buy pass"})
    assert set(by) == {"q"}


def test_cli_runs_both_checks_and_review_by_issue(tmp_path):
    out = tmp_path / "r.json"
    main(["check", "--source", str(FX / "intl_ru.arb"), "--target", str(FX / "intl_kk.arb"),
          "--glossary", str(FX / "glossary_terms.csv"), "--out", str(out), "--checks", "all"])
    types = {d["issue_type"] for d in json.loads(out.read_text("utf-8"))}
    assert "machine_translation_suspect" in types
    main(["check", "--source", str(FX / "intl_ru.arb"), "--target", str(FX / "intl_kk.arb"),
          "--glossary", str(FX / "glossary_terms.csv"), "--out", str(out), "--checks", "glossary"])
    assert all(d["issue_type"] == "glossary_violation" for d in json.loads(out.read_text("utf-8")))
    with pytest.raises(SystemExit):
        main(["check", "--source", "x", "--target", "y", "--out", str(out), "--checks", "nope"])


def test_review_conflicting_fixes_need_qualified_keys(tmp_path):
    rep = [{"key": "k", "issue_type": "machine_translation_suspect", "target": "Іс", "suggestion": "Сандық"},
           {"key": "k", "issue_type": "glossary_violation", "target": "Іс", "suggestion": "Кейс"}]
    (tmp_path / "r.json").write_text(json.dumps(rep, ensure_ascii=False), "utf-8")
    (tmp_path / "d.json").write_text(json.dumps({"k": {"status": "accepted"}}), "utf-8")
    with pytest.raises(SystemExit):
        main(["review", "--report", str(tmp_path / "r.json"), "--decisions", str(tmp_path / "d.json"),
              "--out", str(tmp_path / "o.json")])
    (tmp_path / "d.json").write_text(json.dumps({
        "machine_translation_suspect:k": {"status": "edited", "text": "Кейс"},
        "glossary_violation:k": {"status": "accepted"}}, ensure_ascii=False), "utf-8")
    main(["review", "--report", str(tmp_path / "r.json"), "--decisions", str(tmp_path / "d.json"),
          "--out", str(tmp_path / "o.json")])
    assert json.loads((tmp_path / "o.patch.json").read_text("utf-8")) == {"k": "Кейс"}
