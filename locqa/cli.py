"""locqa CLI (machine_translation_suspect part).

  locqa check  --source intl_ru.arb --target intl_kk.arb --glossary glossary.csv --out report.json
               [--ref-en intl_en.arb] [--llm anthropic|none] [--llm-mode verify|scan]
               [--html report.html] [--merge]
  locqa eval   --report report.json --labels labels.csv   # precision/recall vs answer key
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
from .checks.mt_llm import AnthropicJudge, StaticJudge, Usage


def _load_pricing(path):
    if not path:
        return {}
    return json.loads(Path(path).read_text("utf-8"))


def cmd_check(a) -> int:
    src, tgt = ArbFile(a.source), ArbFile(a.target)
    ref = ArbFile(a.ref_en) if a.ref_en else None
    gloss = gl.load(a.glossary)
    usage = Usage(_load_pricing(a.pricing))
    judge = None
    if a.llm == "anthropic":
        judge = AnthropicJudge(cache_dir=a.cache_dir, batch_size=a.batch_size, usage=usage)
    elif a.llm == "replay":
        judge = StaticJudge(json.loads(Path(a.replay).read_text("utf-8")))
    cfg = mt.MTConfig(llm_mode=a.llm_mode if judge else "none", screen_model=a.screen_model,
                      verify_model=a.verify_model, report_threshold=a.threshold)
    findings, stats = mt.run(src, tgt, judge, cfg, ref, gloss)

    out = Path(a.out)
    existing = []
    if a.merge and out.exists():
        prev = json.loads(out.read_text("utf-8"))
        existing = prev if isinstance(prev, list) else prev.get("findings", [])
        existing = [f for f in existing if f.get("issue_type") != mt.ISSUE]
    payload = existing + [f.to_dict(with_details=not a.no_details) for f in findings]
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    meta = {"stats": stats, "llm_usage": usage.report() if judge else None,
            "config": cfg.__dict__}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    if a.html:
        write_html(Path(a.html), findings, meta)
    print(f"{mt.ISSUE}: {len(findings)} findings ({stats['eligible']} eligible strings) -> {out}",
          file=sys.stderr)
    if judge:
        print(json.dumps(usage.report(), ensure_ascii=False), file=sys.stderr)
    return 0


def write_html(path: Path, findings, meta) -> None:
    e = html.escape
    rows = []
    for f in findings:
        sigs = "".join(f"<li><b>{e(s['name'])}</b> ({s['weight']}): {e(s['reason'])}</li>"
                       for s in f.details.get("signals", []))
        rows.append(f"""<tr class="{f.severity}"><td><code>{e(f.key)}</code></td>
<td><span class="sev">{f.severity}</span><br>{f.confidence:.2f}</td>
<td>{e(f.source)}</td><td>{e(f.target)}</td><td>{e(f.suggestion or '—')}</td>
<td>{e(f.evidence)}<details><summary>signals</summary><ul>{sigs}</ul></details></td></tr>""")
    path.write_text(f"""<!doctype html><meta charset="utf-8"><title>MT suspects</title>
<style>body{{font:14px system-ui;margin:16px}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:6px;vertical-align:top;text-align:left}}
tr.high .sev{{color:#b00020}}tr.medium .sev{{color:#a15c00}}tr.low .sev{{color:#555}}</style>
<h1>machine_translation_suspect — {len(findings)} findings</h1>
<pre>{e(json.dumps(meta['stats'], ensure_ascii=False))}</pre>
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
    c.add_argument("--llm", choices=["none", "anthropic", "replay"], default="none")
    c.add_argument("--replay", help="JSON {key: verdict} for --llm replay")
    c.add_argument("--llm-mode", choices=["verify", "scan"], default="verify")
    c.add_argument("--screen-model", default="claude-haiku-5-5")
    c.add_argument("--verify-model", default="claude-sonnet-5-5")
    c.add_argument("--pricing", help="JSON {model: {input, output, cache_read, cache_write}} USD per 1M tokens")
    c.add_argument("--cache-dir", default=".locqa_cache")
    c.add_argument("--batch-size", type=int, default=15)
    c.add_argument("--threshold", type=float, default=0.5)
    c.add_argument("--merge", action="store_true", help="merge into an existing report.json")
    c.add_argument("--no-details", action="store_true", help="emit only the schema fields")
    c.set_defaults(fn=cmd_check)
    e = sub.add_parser("eval")
    e.add_argument("--report", required=True)
    e.add_argument("--labels", required=True)
    e.set_defaults(fn=cmd_eval)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
