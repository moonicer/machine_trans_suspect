"""Stage 2 of machine_translation_suspect: context-aware LLM judge.

Design choices (why, not just what):
* The LLM is a *judge*, not a translator. It sees one string at a time with
  its key, description, Russian source, English reference, neighbouring keys,
  matching glossary rows and the rule hints, and must point at an exact span.
* Hallucination guard: a verdict whose evidence_span is not a substring of the
  target is discarded. A suggestion that breaks placeholders/ICU, equals the
  target, or glues a suffix to a placeholder is dropped (the finding stays,
  without a fix).
* Deterministic + cheap: temperature 0, batched, system prompt cached, every
  response cached on disk by content hash so re-runs cost nothing.
* Cost is tracked from real usage numbers returned by the API.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .. import icu

PROMPT_VERSION = "mt-judge-v3"

SYSTEM_PROMPT = """You are a senior Kazakh-language editor reviewing the Kazakh UI of an open-world mobile RPG (cars, jobs, factions, families, businesses, police, chat moderation). Translations were made from Russian, sometimes via English or a machine-translation engine.

Task: for each item decide whether the Kazakh text SOUNDS MACHINE-TRANSLATED or literally copied from Russian, i.e. a native Kazakh-speaking player would notice it is a translation.

Flag ("mt") only for these categories:
- calque: word-for-word copy of a Russian/English expression or construction (болып табылады for является, өз кезегінде for в свою очередь, literal idioms).
- russian_grammar: Russian case government or agreement carried over (wrong case after a verb because the Russian verb takes that case; missing possessive suffix after сіздің/біздің; 3rd-person өзінің in a Сіз sentence for Russian свой; Russian words or endings inside Kazakh).
- word_order: non-Kazakh order — possessor after the possessed noun, postposition placed first like a preposition, verb not final in a full sentence, adjective after noun.
- unnatural_register: stiff officialese or dictionary words where players and Kazakh game UIs use something else.
- english_mt: English syntax or Title Case Every Word.

Do NOT flag (answer "ok"):
- established Russian loanwords and game slang players use (варн, мут, бан, фура, ГК/Grand Coins, админ, скин, квест, level-related loans), brands, nicknames, numbers;
- placeholders like {name}, {count}; ICU syntax;
- pure typos or wrong letters (а/ә, к/қ, у/ү ...), terminology mismatches, wrong language, length — other checks handle those;
- short labels that are fine as labels; acceptable but slightly different wording. When in doubt, answer "ok". Precision matters more than recall.

Rule hints are produced by heuristics and MAY BE WRONG; confirm or reject them.

If "mt": give the minimal natural Kazakh rewrite as "suggestion":
- keep every placeholder exactly ({name} stays {name}), keep ICU plural/select structure;
- never attach a suffix directly to a placeholder ({name}-ға is forbidden); rephrase instead (e.g. «Ойыншы {name} ...» or put the placeholder in a position needing no ending);
- keep it no longer than the original for buttons/tabs/titles (judge from the key name);
- follow the glossary rows given; keep the meaning of the Russian source;
- if you are not sure of a good rewrite, set suggestion to null.

Answer with ONLY a JSON array, one object per item, in the same order:
{"id": <id>, "verdict": "mt"|"ok", "categories": [..], "evidence_span": "<exact substring of the Kazakh text>", "explanation": "<one or two English sentences a localization manager understands>", "suggestion": "<kazakh>"|null, "confidence": <0..1>, "severity": "high"|"medium"|"low"}
severity: high = grammar error or meaning distorted; medium = clearly translated-sounding; low = stylistic.
For "ok" items you may leave evidence_span/explanation/suggestion empty."""


@dataclass
class JudgeItem:
    id: int
    key: str
    source: str
    target: str
    ref_en: str = ""
    description: str = ""
    neighbors: list = field(default_factory=list)   # [(key, kk_value)]
    glossary: list = field(default_factory=list)    # [dict]
    hints: list = field(default_factory=list)       # [str]

    def payload(self) -> dict:
        d = {"id": self.id, "key": self.key, "ru": self.source, "kk": self.target}
        if self.ref_en:
            d["en"] = self.ref_en
        if self.description:
            d["description"] = self.description
        if self.neighbors:
            d["neighbors_kk"] = {k: v for k, v in self.neighbors}
        if self.glossary:
            d["glossary"] = self.glossary
        if self.hints:
            d["rule_hints"] = self.hints
        return d


@dataclass
class Verdict:
    is_mt: bool
    confidence: float
    severity: str = "medium"
    categories: list = field(default_factory=list)
    evidence_span: str = ""
    explanation: str = ""
    suggestion: str | None = None
    guard_notes: list = field(default_factory=list)
    model: str = ""


class Usage:
    def __init__(self, pricing: dict | None = None):
        self.lock = threading.Lock()
        self.by_model: dict[str, dict] = {}
        self.pricing = pricing or {}
        self.calls = 0
        self.cache_hits = 0

    def add(self, model: str, u) -> None:
        with self.lock:
            m = self.by_model.setdefault(model, {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
            m["input"] += getattr(u, "input_tokens", 0) or 0
            m["output"] += getattr(u, "output_tokens", 0) or 0
            m["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            m["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
            self.calls += 1

    def report(self) -> dict:
        total, priced = 0.0, True
        out = {"calls": self.calls, "disk_cache_hits": self.cache_hits, "models": {}}
        for model, m in self.by_model.items():
            p = self.pricing.get(model)
            cost = None
            if p and all(p.get(k) is not None for k in ("input", "output")):
                cost = (m["input"] * p["input"] + m["output"] * p["output"]
                        + m["cache_read"] * p.get("cache_read", p["input"])
                        + m["cache_write"] * p.get("cache_write", p["input"])) / 1e6
                total += cost
            else:
                priced = False
            out["models"][model] = {**m, "usd": None if cost is None else round(cost, 4)}
        out["usd_total"] = round(total, 4) if priced and self.by_model else None
        if not priced:
            out["note"] = "set per-model USD/1M-token prices in --pricing to compute cost"
        return out


def _extract_json_array(text: str):
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("no JSON array in model output")
    return json.loads(m.group(0))


_GLUED_SUFFIX = re.compile(r"\}-?(?:ға|ге|қа|ке|да|де|та|те|дан|ден|тан|тен|нан|нен|ның|нің|дың|дің|тың|тің|ны|ні|ды|ді|ты|ті|мен|бен|пен|ның|ға)\b")


def apply_guards(item: JudgeItem, raw: dict, model: str) -> Verdict | None:
    is_mt = str(raw.get("verdict", "")).lower() == "mt"
    try:
        conf = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    v = Verdict(is_mt=is_mt, confidence=conf, model=model,
                severity=raw.get("severity") if raw.get("severity") in ("high", "medium", "low") else "medium",
                categories=[c for c in (raw.get("categories") or []) if isinstance(c, str)],
                evidence_span=(raw.get("evidence_span") or "").strip(),
                explanation=(raw.get("explanation") or "").strip(),
                suggestion=raw.get("suggestion") or None)
    if not is_mt:
        return v
    # 1) evidence must be grounded in the actual string
    if not v.evidence_span or v.evidence_span.lower() not in item.target.lower():
        v.guard_notes.append("evidence_span not found in target -> verdict discarded")
        return None
    # 2) suggestion must be safe to ship
    s = v.suggestion
    if s is not None:
        reason = None
        if not isinstance(s, str) or not s.strip():
            reason = "empty"
        elif s.strip() == item.target.strip():
            reason = "identical to target"
        elif not icu.is_valid(s):
            reason = "invalid ICU syntax"
        elif not icu.same_structure(item.target, s) and not icu.same_structure(item.source, s):
            reason = "placeholder set changed"
        elif _GLUED_SUFFIX.search(s):
            reason = "introduces a suffix glued to a placeholder"
        if reason:
            v.guard_notes.append(f"suggestion dropped: {reason}")
            v.suggestion = None
    return v


class BaseJudge:
    name = "base"

    def judge(self, items: list[JudgeItem], model: str) -> dict[int, Verdict]:
        raise NotImplementedError


class StaticJudge(BaseJudge):
    """Deterministic judge for tests and offline replays: answers come from a
    mapping key -> raw verdict dict (same shape the LLM returns)."""
    name = "static"

    def __init__(self, answers: dict[str, dict]):
        self.answers = answers

    def judge(self, items, model="static"):
        out = {}
        for it in items:
            raw = self.answers.get(it.key, {"verdict": "ok", "confidence": 0.9})
            v = apply_guards(it, {**raw, "id": it.id}, model)
            if v is not None:
                out[it.id] = v
        return out


class AnthropicJudge(BaseJudge):
    name = "anthropic"

    def __init__(self, cache_dir: str | Path = ".locqa_cache", batch_size: int = 15,
                 workers: int = 4, usage: Usage | None = None, max_tokens: int = 4096):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise SystemExit("pip install anthropic  (or run with --llm none)") from e
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set (or run with --llm none)")
        self.client = anthropic.Anthropic()
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size, self.workers, self.max_tokens = batch_size, workers, max_tokens
        self.usage = usage or Usage()

    def _cache_path(self, model: str, item: JudgeItem) -> Path:
        h = hashlib.sha256(json.dumps([PROMPT_VERSION, model, item.payload()], ensure_ascii=False,
                                      sort_keys=True).encode()).hexdigest()[:32]
        return self.cache_dir / f"{h}.json"

    def _call(self, model: str, batch: list[JudgeItem]) -> list[dict]:
        user = ("Review these items. JSON in, JSON array out.\n"
                + json.dumps([b.payload() for b in batch], ensure_ascii=False, indent=1))
        last_err = None
        for _attempt in range(2):
            resp = self.client.messages.create(
                model=model, max_tokens=self.max_tokens, temperature=0,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}])
            self.usage.add(model, resp.usage)
            text = "".join(getattr(c, "text", "") for c in resp.content)
            try:
                arr = _extract_json_array(text)
                return [a for a in arr if isinstance(a, dict)]
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
        raise RuntimeError(f"unparseable model output: {last_err}")

    def judge(self, items, model):
        out: dict[int, Verdict] = {}
        todo: list[JudgeItem] = []
        for it in items:
            p = self._cache_path(model, it)
            if p.exists():
                self.usage.cache_hits += 1
                v = apply_guards(it, json.loads(p.read_text("utf-8")), model)
                if v is not None:
                    out[it.id] = v
            else:
                todo.append(it)
        batches = [todo[i:i + self.batch_size] for i in range(0, len(todo), self.batch_size)]

        def run(batch):
            try:
                raws = self._call(model, batch)
            except Exception as e:  # noqa: BLE001 - one bad batch must not kill the run
                return batch, {}, str(e)
            by_id = {}
            for r in raws:
                try:
                    by_id[int(r.get("id"))] = r
                except (TypeError, ValueError):
                    continue
            return batch, by_id, None

        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for batch, by_id, err in ex.map(run, batches):
                if err:
                    errors.append(err)
                for it in batch:
                    raw = by_id.get(it.id)
                    if raw is None:
                        continue  # missing answer: treated as "no LLM opinion"
                    self._cache_path(model, it).write_text(json.dumps(raw, ensure_ascii=False), "utf-8")
                    v = apply_guards(it, raw, model)
                    if v is not None:
                        out[it.id] = v
        if errors:
            import sys
            print(f"[locqa] LLM: {len(errors)}/{len(batches)} batches failed with model '{model}'. "
                  f"First error: {errors[0][:500]}", file=sys.stderr)
            if len(errors) == len(batches):
                raise SystemExit("every LLM batch failed, so no model verdicts were produced "
                                 "(check the key, the model name, quota and OPENAI_BASE_URL)")
        return out


class OpenAIJudge(AnthropicJudge):
    """Same rubric, batching, guards and disk cache, but through any OpenAI-compatible
    chat-completions endpoint (OpenAI, or a gateway: set OPENAI_BASE_URL)."""
    name = "openai"

    def __init__(self, cache_dir: str | Path = ".locqa_cache", batch_size: int = 15,
                 workers: int = 4, usage: Usage | None = None, max_tokens: int = 4096):
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise SystemExit("pip install openai  (or run with --llm none)") from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set (or run with --llm none)")
        self.client = openai.OpenAI()  # honours OPENAI_API_KEY and OPENAI_BASE_URL
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size, self.workers, self.max_tokens = batch_size, workers, max_tokens
        self.usage = usage or Usage()
        self._no_temperature = False

    def _complete(self, model, messages):
        kw = dict(model=model, messages=messages, max_completion_tokens=self.max_tokens)
        if not self._no_temperature:
            kw["temperature"] = 0
        try:
            return self.client.chat.completions.create(**kw)
        except Exception as e:  # noqa: BLE001
            if "temperature" in str(e) and not self._no_temperature:
                self._no_temperature = True   # reasoning models only accept the default
                kw.pop("temperature", None)
                return self.client.chat.completions.create(**kw)
            raise

    def _call(self, model, batch):
        user = ("Review these items. JSON in, JSON array out.\n"
                + json.dumps([b.payload() for b in batch], ensure_ascii=False, indent=1))
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        last_err = None
        for _attempt in range(2):
            resp = self._complete(model, msgs)
            u = resp.usage
            self.usage.add(model, type("U", (), {
                "input_tokens": getattr(u, "prompt_tokens", 0),
                "output_tokens": getattr(u, "completion_tokens", 0)})())
            text = resp.choices[0].message.content or ""
            try:
                return [a for a in _extract_json_array(text) if isinstance(a, dict)]
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
        raise RuntimeError(f"unparseable model output: {last_err}")
