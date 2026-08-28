from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class CAVDBoundary:
    w: float
    b: float
    sigma: float
    k: float = 1.0

    def threshold(self, seq_len: int) -> float:
        return self.w * float(seq_len) + self.b + self.k * self.sigma


@dataclass(frozen=True)
class CAVDScore:
    seq_len: int
    avr: float
    threshold: float
    flagged: bool
    layer_idx: int


@dataclass(frozen=True)
class GuardOutput:
    harmful: Optional[bool]
    text: str
    view_name: str


@dataclass(frozen=True)
class CAVDDecision:
    harmful: Optional[bool]
    cavd: CAVDScore
    candidate_views: List[Tuple[str, str]]
    evaluated_outputs: List[GuardOutput]
