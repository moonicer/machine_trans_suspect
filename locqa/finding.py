from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Finding:
    """One issue, in the exact report.json schema from the brief, plus an
    optional `details` block (machine-readable signals) for explainability."""
    key: str
    issue_type: str
    severity: str            # high | medium | low
    source: str
    target: str
    evidence: str
    suggestion: str | None
    confidence: float
    reviewed_by_human: bool = False
    details: dict = field(default_factory=dict)

    def to_dict(self, with_details: bool = True) -> dict:
        d = asdict(self)
        d["confidence"] = round(float(self.confidence), 3)
        if not with_details:
            d.pop("details", None)
        return d
