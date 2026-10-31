import json

import pytest

from locqa.arb import ArbFile
from locqa.cli import main

SRC = """{
  "@@locale": "ru",
  "a": "Купить {item}",
  "b": "Привет",
  "@b": {
    "description": "greeting"
  },
  "@@comment": "X",
  "c": "Цена {price}",
  "d": "Конец"
}
"""

# duplicate key "a" (last wins), repeated @@comment, multi-line @b, last entry without comma
TGT = """{
  "@@locale": "kk",
  "a": "Сатып алу {item}",
  "b": "Сәлем",
  "@b": {
    "description": "greeting"
  },
  "@@comment": "X",
  "c": "Баға {price}",
  "a": "Сатып алу {item}",
  "@@comment": "Y",
  "d": "Соңы"
}
"""


def report(path, items):
    findings = [{"key": k, "issue_type": "machine_translation_suspect", "severity": "high", "source": "",
                 "target": "", "evidence": "", "suggestion": s, "confidence": 0.9, "reviewed_by_human": r,
                 "details": {"review": {"reviewer": "native"}}} for k, s, r in items]
    path.write_text(json.dumps(findings, ensure_ascii=False), "utf-8")


def run(tmp_path, items, *extra):
    (tmp_path / "ru.arb").write_text(SRC, "utf-8")
    (tmp_path / "kk.arb").write_text(TGT, "utf-8")
    report(tmp_path / "r.json", items)
    main(["apply", "--source", str(tmp_path / "ru.arb"), "--target", str(tmp_path / "kk.arb"),
          "--report", str(tmp_path / "r.json"), "--out", str(tmp_path / "fixed.arb"), *extra])
    return (tmp_path / "fixed.arb").read_text("utf-8")


def test_apply_edits_in_place_and_flags(tmp_path):
    out = run(tmp_path, [("a", "{item} сатып алу", True), ("b", "Сәлеметсіз бе", True), ("d", "Аяқтау", True)])
    data = json.loads(out)
    assert data["a"] == "{item} сатып алу" and data["b"] == "Сәлеметсіз бе" and data["d"] == "Аяқтау"
    assert data["@a"]["x-locqa-reviewed"] is True and data["@a"]["x-locqa-reviewer"] == "native"
    assert data["@b"]["description"] == "greeting"          # existing metadata kept and merged
    assert data["@b"]["x-locqa-reviewed"] is True
    assert "@c" not in data                                 # untouched key gets no flag
    assert out.count('"@@comment"') == 2                    # repeated markers survive (no JSON round trip)
    assert out.count('"a": "{item} сатып алу"') == 2        # duplicate line patched too
    lines = out.split("\n")
    assert lines[-3].strip().startswith('"@d"') and not lines[-3].rstrip().endswith(",")  # last entry still valid


def test_unreviewed_not_applied_by_default(tmp_path):
    out = run(tmp_path, [("c", "Баға: {price}", False)])
    assert json.loads(out)["c"] == "Баға {price}"
    assert "x-locqa" not in out


def test_include_unreviewed_flags_false(tmp_path):
    out = run(tmp_path, [("c", "Баға: {price}", False)], "--include-unreviewed")
    data = json.loads(out)
    assert data["c"] == "Баға: {price}" and data["@c"]["x-locqa-reviewed"] is False


def test_placeholder_loss_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, [("c", "Баға", True)])
    assert not (tmp_path / "fixed.arb").exists()


def test_conflicting_fixes_for_one_key(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, [("a", "{item} алу", True), ("a", "{item} сатып алу", True)])


def test_sidecar_review_log(tmp_path):
    run(tmp_path, [("b", "Сәлеметсіз бе", True)])
    log = json.loads((tmp_path / "fixed.review.json").read_text("utf-8"))
    assert log["b"]["before"] == "Сәлем" and log["b"]["reviewed_by_human"] is True


def test_fixed_file_loads_with_arbfile(tmp_path):
    run(tmp_path, [("b", "Сәлеметсіз бе", True)])
    assert ArbFile(tmp_path / "fixed.arb").get("b").value == "Сәлеметсіз бе"


def test_html_shows_review_status(tmp_path):
    report(tmp_path / "r.json", [("b", "Сәлеметсіз бе", True), ("c", "Баға: {price}", False)])
    main(["html", "--report", str(tmp_path / "r.json"), "--out", str(tmp_path / "r.html")])
    h = (tmp_path / "r.html").read_text("utf-8")
    assert "1 reviewed by a human" in h and "pending" in h
