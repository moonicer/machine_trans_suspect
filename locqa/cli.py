"""locqa CLI.

  locqa check  --source intl_ru.arb --target intl_kk.arb --glossary glossary.csv --out report.json
               [--checks all|mt,glossary] [--ref-en intl_en.arb]
               [--llm anthropic|openai|none|replay] [--llm-mode verify|scan]
               [--html report.html] [--merge]
  locqa eval   --report report.json --labels labels.csv   # precision/recall vs answer key
  locqa review --report report.json --decisions decisions.json --out report.reviewed.json
  locqa combine --reports mt.json glossary.json manual.json --out report.json   # + report.patch.json
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

from . import glossary as gl
from .arb import ArbFile
from .checks import machine_translation as mt
from .checks.mt_llm import AnthropicJudge, OpenAIJudge, StaticJudge, Usage


def _load_pricing(path):
    if not path:
        return {}
    return json.loads(Path(path).read_text("utf-8"))


CHECKS = ("machine_translation_suspect", "glossary_violation")
ALIASES = {"mt": "machine_translation_suspect", "glossary": "glossary_violation"}


def _selected(spec: str) -> list[str]:
    if spec in ("", "all"):
        return list(CHECKS)
    out = []
    for x in spec.split(","):
        x = ALIASES.get(x.strip(), x.strip())
        if x not in CHECKS:
            raise SystemExit(f"unknown check {x!r}; available: {', '.join(CHECKS)}")
        out.append(x)
    return out


def cmd_check(a) -> int:
    from .checks import glossary_violation as gv
    checks = _selected(a.checks)
    src, tgt = ArbFile(a.source), ArbFile(a.target)
    ref = ArbFile(a.ref_en) if a.ref_en else None
    terms = gl.load_terms(a.glossary)
    findings, meta = [], {"stats": {}, "llm_usage": None, "config": {}}

    if "glossary_violation" in checks:
        if not terms:
            print("glossary_violation: no --glossary given, skipped", file=sys.stderr)
        else:
            f, st = gv.run(src, tgt, terms, ref)
            findings += f; meta["stats"][gv.ISSUE] = st
            print(f"{gv.ISSUE}: {len(f)} findings ({st['keys_with_terms']} keys contain glossary terms)",
                  file=sys.stderr)

    if "machine_translation_suspect" in checks:
        usage = Usage(_load_pricing(a.pricing))
        judge = None
        if a.llm == "anthropic":
            judge = AnthropicJudge(cache_dir=a.cache_dir, batch_size=a.batch_size, usage=usage, max_tokens=a.max_tokens)
        elif a.llm == "openai":
            if not (a.screen_model and a.verify_model):
                raise SystemExit("--llm openai needs --screen-model and --verify-model (e.g. the same cheap model for both)")
            judge = OpenAIJudge(cache_dir=a.cache_dir, batch_size=a.batch_size, usage=usage, max_tokens=a.max_tokens)
        elif a.llm == "replay":
            judge = StaticJudge(json.loads(Path(a.replay).read_text("utf-8")))
        cfg = mt.MTConfig(llm_mode=a.llm_mode if judge else "none",
                          screen_model=a.screen_model or "claude-haiku-5-5",
                          verify_model=a.verify_model or "claude-sonnet-5-5", report_threshold=a.threshold,
                          min_scan_words=a.min_scan_words)
        f, st = mt.run(src, tgt, judge, cfg, ref, [t.as_dict() for t in terms])
        findings += f; meta["stats"][mt.ISSUE] = st; meta["config"][mt.ISSUE] = cfg.__dict__
        if judge:
            meta["llm_usage"] = usage.report()
            print(json.dumps(usage.report(), ensure_ascii=False), file=sys.stderr)
        print(f"{mt.ISSUE}: {len(f)} findings ({st['eligible']} eligible strings)", file=sys.stderr)

    out = Path(a.out)
    existing = []
    if a.merge and out.exists():
        prev = json.loads(out.read_text("utf-8"))
        existing = prev if isinstance(prev, list) else prev.get("findings", [])
        existing = [f for f in existing if f.get("issue_type") not in checks]
    payload = existing + [f.to_dict(with_details=not a.no_details) for f in findings]
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    if a.html:
        write_html(Path(a.html), findings, meta)
    print(f"-> {out}", file=sys.stderr)
    return 0


def write_html(path: Path, findings, meta) -> None:
    e = html.escape
    rows = []
    for f in findings:
        sigs = "".join(f"<li><b>{e(s['name'])}</b> ({s['weight']}): {e(s['reason'])}</li>"
                       for s in f.details.get("signals", []))
        sigs += "".join(f"<li><b>{e(t['kind'])}</b> {e(t['term'])} → {e(t['approved'])}</li>"
                        for t in f.details.get("terms", []))
        rows.append(f"""<tr class="{f.severity}"><td><code>{e(f.key)}</code><br><small>{e(f.issue_type)}</small></td>
<td><span class="sev">{f.severity}</span><br>{f.confidence:.2f}</td>
<td>{e(f.source)}</td><td>{e(f.target)}</td><td>{e(f.suggestion or '—')}</td>
<td>{e(f.evidence)}<details><summary>signals</summary><ul>{sigs}</ul></details></td></tr>""")
    path.write_text(f"""<!doctype html><meta charset="utf-8"><title>LocQA report</title>
<style>body{{font:14px system-ui;margin:16px}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:6px;vertical-align:top;text-align:left}}
tr.high .sev{{color:#b00020}}tr.medium .sev{{color:#a15c00}}tr.low .sev{{color:#555}}</style>
<h1>LocQA — {len(findings)} findings</h1>
<pre>{e(json.dumps(meta.get('stats', {}), ensure_ascii=False))}</pre>
<table><tr><th>key</th><th>severity / conf</th><th>ru</th><th>kk</th><th>suggestion</th><th>evidence</th></tr>
{''.join(rows)}</table>""", "utf-8")


def cmd_eval(a) -> int:
    """labels.csv: key,issue_type  (answer key or native-speaker labels).
    Prints precision/recall of MT findings at several confidence cut-offs."""
    rep = json.loads(Path(a.report).read_text("utf-8"))
    rep = rep if isinstance(rep, list) else rep.get("findings", [])
    pred = [(f["key"], f["confidence"]) for f in rep if f["issue_type"] == mt.ISSUE]
    with open(a.labels, encoding="utf-8-sig", newline="") as fh:
        gold = {r["key"] for r in csv.DictReader(fh) if r.get("issue_type", mt.ISSUE) == mt.ISSUE}
    print("threshold  n_pred  precision  recall")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        p = {k for k, c in pred if c >= t}
        tp = len(p & gold)
        prec = tp / len(p) if p else 0.0
        rec = tp / len(gold) if gold else 0.0
        print(f"{t:9.1f}  {len(p):6d}  {prec:9.3f}  {rec:6.3f}")
    return 0


def cmd_review(a) -> int:
    """Apply native-speaker decisions to a report.

    decisions.json: {key or "issue_type:key": {"status": "accepted" | "edited" | "rejected",
                           "text": "<final kk, for edited>", "note": "...", "reviewer": "..."}}
    accepted -> suggestion kept, reviewed_by_human=true
    edited   -> suggestion replaced by the reviewer's text (ICU/placeholders re-validated)
    rejected -> finding removed (reviewer says it is not an error)
    Findings without a decision stay reviewed_by_human=false.
    Also writes <out>.patch.json {key: final_text} for building intl_kk_fixed.arb.
    """
    from . import icu
    rep = json.loads(Path(a.report).read_text("utf-8"))
    dec = json.loads(Path(a.decisions).read_text("utf-8"))
    out, patch, counts = [], {}, {"accepted": 0, "edited": 0, "rejected": 0, "pending": 0}
    for f in rep:
        d = dec.get(f"{f.get('issue_type')}:{f['key']}") or dec.get(f["key"])
        if not d:
            counts["pending"] += 1
            out.append(f); continue
        st = d.get("status")
        if st == "rejected":
            counts["rejected"] += 1; continue
        if st == "edited":
            text = d["text"]
            if not icu.is_valid(text) or not icu.same_structure(f["target"], text):
                raise SystemExit(f"{f['key']}: reviewer text breaks ICU/placeholders")
            f["suggestion"] = text
        elif st != "accepted":
            raise SystemExit(f"{f['key']}: unknown status {st!r}")
        f["reviewed_by_human"] = True
        if d.get("note") or d.get("reviewer"):
            f.setdefault("details", {})["review"] = {k: d[k] for k in ("note", "reviewer") if d.get(k)}
        counts[st] += 1
        if f.get("suggestion"):
            if f["key"] in patch and patch[f["key"]] != f["suggestion"]:
                raise SystemExit(f"{f['key']}: two reviewed fixes disagree; decide with 'issue_type:key'")
            patch[f["key"]] = f["suggestion"]
        out.append(f)
    o = Path(a.out)
    o.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    o.with_suffix(".patch.json").write_text(json.dumps(patch, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(counts), file=sys.stderr)
    return 0


def cmd_combine(a) -> int:
    """Concatenate reports from several checks and build ONE patch:
    {key: final_text} from reviewed findings. Fails if two reviewed findings
    for the same key propose different text (the fix must be a single string
    that resolves every issue of that key)."""
    allf, patch, by_type = [], {}, {}
    for r in a.reports:
        data = json.loads(Path(r).read_text("utf-8"))
        allf += data if isinstance(data, list) else data.get("findings", [])
    for f in allf:
        by_type[f["issue_type"]] = by_type.get(f["issue_type"], 0) + 1
        if f.get("reviewed_by_human") and f.get("suggestion"):
            if f["key"] in patch and patch[f["key"]] != f["suggestion"]:
                raise SystemExit(f"{f['key']}: reviewed fixes disagree between issue types")
            patch[f["key"]] = f["suggestion"]
    o = Path(a.out)
    o.write_text(json.dumps(allf, ensure_ascii=False, indent=2), "utf-8")
    o.with_suffix(".patch.json").write_text(json.dumps(patch, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"findings": len(allf), "by_type": by_type,
                      "reviewed": sum(1 for f in allf if f.get("reviewed_by_human")),
                      "patched_keys": len(patch)}, ensure_ascii=False), file=sys.stderr)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="locqa")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--source", required=True)
    c.add_argument("--target", required=True)
    c.add_argument("--glossary")
    c.add_argument("--out", required=True)
    c.add_argument("--ref-en", help="optional third-language reference ARB (e.g. intl_en.arb)")
    c.add_argument("--html")
    c.add_argument("--checks", default="all", help="all, or comma list: mt, glossary")
    c.add_argument("--llm", choices=["none", "anthropic", "openai", "replay"], default="none")
    c.add_argument("--replay", help="JSON {key: verdict} for --llm replay")
    c.add_argument("--llm-mode", choices=["verify", "scan"], default="verify")
    c.add_argument("--screen-model")
    c.add_argument("--verify-model")
    c.add_argument("--pricing", help="JSON {model: {input, output, cache_read, cache_write}} USD per 1M tokens")
    c.add_argument("--cache-dir", default=".locqa_cache")
    c.add_argument("--batch-size", type=int, default=15)
    c.add_argument("--max-tokens", type=int, default=4096, help="output cap per LLM call; raise it for reasoning models")
    c.add_argument("--threshold", type=float, default=0.5)
    c.add_argument("--min-scan-words", type=int, default=3)
    c.add_argument("--merge", action="store_true", help="merge into an existing report.json")
    c.add_argument("--no-details", action="store_true", help="emit only the schema fields")
    c.set_defaults(fn=cmd_check)
    e = sub.add_parser("eval")
    e.add_argument("--report", required=True)
    e.add_argument("--labels", required=True)
    e.set_defaults(fn=cmd_eval)
    r = sub.add_parser("review")
    r.add_argument("--report", required=True)
    r.add_argument("--decisions", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_review)
    cb = sub.add_parser("combine")
    cb.add_argument("--reports", nargs="+", required=True)
    cb.add_argument("--out", required=True)
    cb.set_defaults(fn=cmd_combine)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
