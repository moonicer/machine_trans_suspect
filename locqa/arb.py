"""Minimal, dependency-free ARB loader.

Works with any valid ARB: translatable keys are those not starting with '@';
'@key' objects carry description / placeholders metadata.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ArbEntry:
    key: str
    value: str
    description: str = ""
    placeholders: dict = field(default_factory=dict)


class ArbFile:
    def __init__(self, path: str | Path | None = None, data: dict | None = None):
        if data is None:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh, object_pairs_hook=OrderedDict)
        self.path = Path(path) if path else None
        self.raw: OrderedDict = data
        self.locale = data.get("@@locale", "")

    def keys(self) -> list[str]:
        return [k for k, v in self.raw.items() if not k.startswith("@") and isinstance(v, str)]

    def get(self, key: str) -> ArbEntry | None:
        v = self.raw.get(key)
        if not isinstance(v, str):
            return None
        meta = self.raw.get("@" + key) or {}
        return ArbEntry(key, v, meta.get("description", "") or "", meta.get("placeholders", {}) or {})

    def __iter__(self):
        for k in self.keys():
            yield self.get(k)

    def neighbors(self, key: str, n: int = 2) -> list[ArbEntry]:
        """Keys adjacent in file order: context for keys without descriptions."""
        ks = self.keys()
        try:
            i = ks.index(key)
        except ValueError:
            return []
        idx = [j for j in range(i - n, i + n + 1) if j != i and 0 <= j < len(ks)]
        return [self.get(ks[j]) for j in idx]
