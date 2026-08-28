from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import CAVDBoundary


def fit_cavd_boundary(seq_lens: Sequence[int], avr_values: Sequence[float], k: float = 1.0) -> CAVDBoundary:
    if len(seq_lens) != len(avr_values):
        raise ValueError("seq_lens and avr_values must have the same length")
    if len(seq_lens) < 2:
        raise ValueError("at least two calibration prompts are required")
    L = np.asarray(seq_lens, dtype=np.float64)
    V = np.asarray(avr_values, dtype=np.float64)
    w, b = np.polyfit(L, V, 1)
    sigma = float(np.std(V - (w * L + b)))
    return CAVDBoundary(float(w), float(b), sigma, float(k))
