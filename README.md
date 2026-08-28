# CAVD-Guard Core

Minimal core implementation for CAVD-Guard. This release contains no datasets or experiment outputs.

## Layout

- `cavd_guard/scoring.py`: Attention-Void Ratio.
- `cavd_guard/calibration.py`: length-conditioned CAVD boundary.
- `cavd_guard/canonical_views.py`: deterministic canonical-view generation with explicit `t0`, `t1`, and `t2` tiers.
- `cavd_guard/guards.py`: guard-model wrapper and CAVD-Guard routing.
- `cavd_guard/constants.py`: fixed normalization patterns and the bounded character-repair vocabulary.

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

The default CAVD settings match the paper: terminal layer, `Wlocal=5`, `Wprefix=30`, `alpha=0.1`, and `k=1`.

`decision.candidate_views` stores all generated canonical views. `decision.evaluated_outputs` stores only the views actually inspected before early stopping.

The reported `seq_len` is the token length of the guard-rendered input used for the CAVD prefill pass. By default `drop_first_token=True`, matching the reproduction code's removal of the initial special/template token before computing AVR.

`RISKY_CANONICAL_TERMS` is used only by bounded character-edit repair in canonical-view generation. It is not used as a harmfulness classifier; the guard model makes the final harmfulness decision.
