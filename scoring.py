from __future__ import annotations

import torch


def attention_void_ratio(
    attention: torch.Tensor,
    local_window: int = 5,
    prefix_window: int = 30,
    alpha: float = 0.1,
) -> float:
    attn = attention.to(dtype=torch.float32)
    seq_len = int(attn.shape[-1])
    if seq_len <= local_window + prefix_window:
        return 0.0
    device = attn.device
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    dist = row_idx - col_idx
    measured_mask = dist > local_window
    reference_mask = measured_mask & (col_idx >= prefix_window)
    measured = attn[measured_mask]
    reference = attn[reference_mask]
    if measured.numel() == 0 or reference.numel() == 0:
        return 0.0
    mu = reference.mean()
    if float(mu.item()) <= 1e-9:
        return 0.0
    return float((measured < alpha * mu).float().mean().item())
