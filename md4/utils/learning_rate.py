"""Learning rate scheduling utilities using numpy."""

import numpy as np
from typing import Any
from absl import logging


def cosine_decay(
    lr: Any, current_step: Any, total_steps: Any
) -> Any:
    """Cosine decay that accepts Python scalars or numpy arrays.
    
    Args:
        lr: Base learning rate
        current_step: Current training step
        total_steps: Total number of training steps
        
    Returns:
        Decayed learning rate value
    """
    current_step = np.asarray(current_step, dtype=np.float32)
    total_steps = np.maximum(1.0, np.asarray(total_steps, dtype=np.float32))
    lr = np.asarray(lr, dtype=np.float32)
    ratio = np.maximum(0.0, current_step / total_steps)
    mult = 0.5 * (1.0 + np.cos(np.pi * ratio))
    return mult * lr


def get_learning_rate(
    step: int,
    *,
    base_learning_rate: float,
    num_steps: int,
    warmup_steps: int | None = None,
    schedule_type: str = "cosine",
) -> Any:
    """Learning rate schedule helper.

    Supports:
      - "cosine": single cosine decay with (optional) warmup.
      - "constant": constant LR after warmup.
      - "cyclic_cosine": cosine annealing with warm restarts (cyclic cosine).

    The cyclic variant optionally allows inline parameter overrides encoded in
    the schedule_type string, e.g.:
        "cyclic_cosine;cycle_length=1000;min_lr=1e-5"
    Parameters (all optional) for cyclic_cosine:
        cycle_length:   Number of steps per cycle (default: num_steps // 10, >=1).
        min_lr:         Minimum LR at the end of each cycle (default: 0.0).
        decay_factor:   Multiplicative factor applied to base LR after each cycle
                        restart (default: 1.0, i.e., no decay of peaks).
    Warmup (if warmup_steps provided) is applied only to the very beginning of
    training (first warmup_steps) scaling the scheduled LR linearly.
    
    Args:
        step: Current training step
        base_learning_rate: Base learning rate
        num_steps: Total number of training steps
        warmup_steps: Number of warmup steps (optional)
        schedule_type: Type of learning rate schedule
        
    Returns:
        Learning rate for the current step
    """
    logging.info(
        "get_learning_rate(step=%s, base_learning_rate=%s, num_steps=%s, schedule_type=%s)",
        step,
        base_learning_rate,
        num_steps,
        schedule_type,
    )

    # Handle warmup (gracefully if warmup_steps is None or 0).
    if warmup_steps is None or warmup_steps <= 0:
        warmup = 1.0
        effective_step = step
        effective_total = num_steps
    else:
        warmup = np.minimum(1.0, step / warmup_steps)
        effective_step = np.maximum(0, step - warmup_steps)
        effective_total = np.maximum(1, num_steps - warmup_steps)

    # Allow parameter overrides for cyclic cosine via semi-colon separated kv pairs.
    schedule_base = schedule_type.split(";")[0]
    extra_params = schedule_type.split(";")[1:]
    parsed: dict[str, str] = {}
    for kv in extra_params:
        if "=" in kv:
            k, v = kv.split("=", 1)
            parsed[k.strip()] = v.strip()

    if schedule_base == "cosine":
        lr = cosine_decay(base_learning_rate, effective_step, effective_total)
    elif schedule_base == "constant":
        lr = base_learning_rate
    elif schedule_base == "cyclic_cosine":
        # Derive cycle_length (at least 1) and other params.
        default_cycle = max(1, num_steps // 10)
        try:
            cycle_length = int(parsed.get("cycle_length", default_cycle))
        except ValueError:  # Fall back to default if parsing fails.
            cycle_length = default_cycle
        cycle_length = max(1, cycle_length)

        try:
            min_lr = float(parsed.get("min_lr", 0.0))
        except ValueError:
            min_lr = 0.0
        try:
            decay_factor = float(parsed.get("decay_factor", 1.0))
        except ValueError:
            decay_factor = 1.0

        # Position within current cycle (after warmup region).
        cycle_index = np.floor_divide(effective_step, cycle_length)
        pos_in_cycle = np.mod(effective_step, cycle_length)

        # Optionally decay the peak LR each cycle.
        peak_lr = base_learning_rate * (decay_factor**cycle_index)

        # Cosine within the cycle from peak_lr down to min_lr.
        cosine_ratio = pos_in_cycle / cycle_length
        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + np.cos(np.pi * cosine_ratio))
    else:
        raise NotImplementedError(f"Unknown schedule type: {schedule_type}")

    return np.asarray(lr * warmup, dtype=np.float32)