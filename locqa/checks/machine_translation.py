"""machine_translation_suspect — orchestrator.

Pipeline
  0. eligibility   skip empty / identical / mostly-Russian strings (owned by
                   untranslated, same_as_source, wrong_language checks)
  1. rules         offline Kazakh-specific signals, noisy-OR -> p_rule
  2. LLM screen    (mode=scan) cheap model over multi-word strings rules missed
  3. LLM verify    stronger model over rule candidates + screen positives;
                   produces span-grounded evidence and a guarded suggestion
  4. fusion        confidence = 1-(1-p_rule)(1-p_llm); LLM "ok" vetoes weak
                   rule-only hits; strong rule hits survive with lowered
                   confidence and an explicit "LLM disagrees" note
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import glossary as gl
from .. import icu
from ..arb import ArbFile
from ..finding import Finding
from ..text_kk import PH, words
from .mt_llm import BaseJudge, JudgeItem, Verdict
from .mt_rules import eligible, noisy_or, rule_signals

ISSUE = "machine_translation_suspect"
SEV_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass
class MTConfig:
    llm_mode: str = "verify"          # none | verify | scan
    screen_model: str = "claude-haiku-5-5"
    verify_model: str = "claude-sonnet-5-5"
    candidate_threshold: float = 0.30  # p_rule needed to be sent to verify
    report_threshold_rules_only: float = 0.55
    strong_rule_threshold: float = 0.75  # survives an LLM "ok"
    min_scan_words: int = 3
    report_threshold: float = 0.50
    keys: list = field(default_factory=list)  # optional subset


def _n_words(s: str) -> int:
    return sum(1 for w in words(" ".join(icu.text_segments(s))) if w != PH)


def _severity(sigs, verdict: Verdict | None) -> str:
    sev = "low"
    for s in sigs:
        if SEV_RANK[s.severity] > SEV_RANK[sev]:
            sev = s.severity
    if verdict and verdict.is_mt and SEV_RANK.get(verdict.severity, 1) > SEV_RANK[sev]:
        sev = verdict.severity
    return sev


def _evidence(sigs, verdict: Verdict | None, llm_disagrees: bool) -> str:
    parts = []
    if verdict and verdict.is_mt and verdict.explanation:
        cats = f" [{', '.join(verdict.categories)}]" if verdict.categories else ""
        parts.append(f"{verdict.explanation}{cats} — span: «{verdict.evidence_span}»")
    for s in sorted(sigs, key=lambda x: -x.weight)[:3]:
        parts.append(s.reason)
    if llm_disagrees:
        parts.append("note: LLM reviewer judged the string acceptable; kept because the rule evidence is strong")
    return " | ".join(parts)


def run(source: ArbFile, target: ArbFile, judge: BaseJudge | None = None,
        cfg: MTConfig | None = None, ref_en: ArbFile | None = None,
        glossary: list | None = None) -> tuple[list[Finding], dict]:
    cfg = cfg or MTConfig()
    glossary = glossary or []
    stats = {"eligible": 0, "skipped": {}, "rule_candidates": 0, "screened": 0,
             "screen_positive": 0, "verified": 0, "reported": 0}

    rows = []  # (key, src, tgt, en, sigs, p_rule)
    keys = cfg.keys or source.keys()
    for key in keys:
        se, te = source.get(key), target.get(key)
        if se is None or te is None:
            continue
        ok, why = eligible(se.value, te.value)
        if not ok:
            stats["skipped"][why] = stats["skipped"].get(why, 0) + 1
            continue
        stats["eligible"] += 1
        en = ref_en.get(key).value if ref_en and ref_en.get(key) else None
        sigs = rule_signals(se.value, te.value, en)
        rows.append((key, se, te, en, sigs, noisy_or(s.weight for s in sigs)))

    def item(i, key, se, te, en, sigs):
        return JudgeItem(
            id=i, key=key, source=se.value, target=te.value, ref_en=en or "",
            description=se.description or te.description,
            neighbors=[(n.key, n.value) for n in target.neighbors(key, 2) if n],
            glossary=gl.relevant(glossary, se.value, te.value),
            hints=[f"{s.name}: {s.reason}" for s in sigs])

    verdicts: dict[str, Verdict] = {}
    use_llm = judge is not None and cfg.llm_mode != "none"
    if use_llm:
        cand = {r[0] for r in rows if r[5] >= cfg.candidate_threshold}
        stats["rule_candidates"] = len(cand)
        if cfg.llm_mode == "scan":
            pool = [r for r in rows if r[0] not in cand and _n_words(r[2].value) >= cfg.min_scan_words]
            items = [item(i, *r[:5]) for i, r in enumerate(pool)]
            stats["screened"] = len(items)
            if cfg.screen_model == cfg.verify_model:
                for it, v in ((items[i], v) for i, v in judge.judge(items, cfg.screen_model).items()):
                    verdicts[it.key] = v
            else:
                res = judge.judge(items, cfg.screen_model)
                for i, v in res.items():
                    if v.is_mt:
                        cand.add(items[i].key)
                stats["screen_positive"] = sum(1 for v in res.values() if v.is_mt)
        vrows = [r for r in rows if r[0] in cand and r[0] not in verdicts]
        vitems = [item(i, *r[:5]) for i, r in enumerate(vrows)]
        stats["verified"] = len(vitems)
        for i, v in judge.judge(vitems, cfg.verify_model).items():
            verdicts[vitems[i].key] = v

    findings: list[Finding] = []
    for key, se, te, en, sigs, p_rule in rows:
        v = verdicts.get(key)
        llm_disagrees = False
        if v is not None and v.is_mt:
            conf = 1 - (1 - p_rule) * (1 - v.confidence)
        elif v is not None and not v.is_mt:
            if p_rule < cfg.strong_rule_threshold:
                continue
            llm_disagrees = True
            conf = p_rule * 0.6
        else:
            if not sigs:
                continue
            thr = cfg.report_threshold_rules_only if not use_llm else cfg.report_threshold
            if p_rule < thr:
                continue
            conf = p_rule
        if conf < cfg.report_threshold and not llm_disagrees:
            continue
        findings.append(Finding(
            key=key, issue_type=ISSUE, severity=_severity(sigs, v), source=se.value,
            target=te.value, evidence=_evidence(sigs, v, llm_disagrees),
            suggestion=v.suggestion if (v and v.is_mt) else None,
            confidence=round(conf, 3), reviewed_by_human=False,
            details={"rule_score": round(p_rule, 3), "signals": [s.to_dict() for s in sigs],
                     "llm": None if v is None else {
                         "model": v.model, "verdict": "mt" if v.is_mt else "ok",
                         "confidence": v.confidence, "categories": v.categories,
                         "evidence_span": v.evidence_span, "guard_notes": v.guard_notes}}))
    findings.sort(key=lambda f: (-SEV_RANK[f.severity], -f.confidence, f.key))
    stats["reported"] = len(findings)
    return findings, stats
