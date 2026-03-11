"""GPU-aware hyperparameter scaling for A100 and other CUDA devices."""

import torch


def _is_a100() -> bool:
    """Detect A100 (40GB or 80GB)."""
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name(0).upper()
    return "A100" in name


def gpu_batch_size(base: int) -> int:
    """Scale batch size for GPU throughput.

    - A100: 4x, cap 2048
    - Other CUDA: 2x, cap 512
    - CPU: base unchanged
    """
    if not torch.cuda.is_available():
        return base
    if _is_a100():
        return min(base * 4, 2048)
    return min(base * 2, 512)


def gpu_hidden_dim(base: int) -> int:
    """Scale hidden dim for GPU. A100 can use larger networks."""
    if not torch.cuda.is_available():
        return base
    if _is_a100():
        return min(int(base * 1.5), 512)
    return base


def gpu_info() -> str:
    """Return GPU info string for logging."""
    if not torch.cuda.is_available():
        return "CPU"
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return f"{name} ({vram_gb:.1f} GB)"
