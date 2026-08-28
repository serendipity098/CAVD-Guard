# CAVD-Guard

Core implementation of **CAVD-Guard: Defending Against Token Fragmentation Jailbreaks via Attention Footprints** (EMNLP 2026).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from cavd_guard import CAVDGuard, GuardModel

guard = GuardModel.from_pretrained(
    "path/to/guard",
    model_type="wildguard",
)

cavd_guard = CAVDGuard.calibrate(
    guard,
    calibration_prompts,
    k=1.0,
)

decision = cavd_guard.classify(user_prompt)
print(decision.harmful, decision.cavd.flagged, decision.cavd.avr)
```

## Core Modules

- `scoring.py`: Attention-Void Ratio computation.
- `calibration.py`: length-conditioned CAVD calibration.
- `canonical_views.py`: deterministic canonical-view generation.
- `guards.py`: guard-model wrapper and CAVD-Guard routing.
- `constants.py`: normalization patterns and bounded character-repair vocabulary.
