#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

"""
Useful learning rate schedulers
Warmup
CosineAnnealingLRWarmup
"""
import typing as t
import torch
import math
import functools


def _cosine_decay_warmup(
    iteration: int, warmup_iterations: int, total_iterations: int, min_multiplier: float = 1e-3
) -> float:
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.0
    """
    if iteration <= warmup_iterations:
        multiplier = max(iteration / warmup_iterations, min_multiplier)
    else:
        multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier


def _constant_warmup(iteration: int, warmup_iterations: int) -> float:
    """
    Linear warmup from 0 --> 1.0, then constant
    """
    multiplier = 1.0
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    return multiplier


def CosineAnnealingLRWarmup(optimizer: torch.optim.Optimizer, T_max: int, T_warmup: int) -> t.Any:
    _decay_func = functools.partial(
        _cosine_decay_warmup, warmup_iterations=T_warmup, total_iterations=T_max
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler


def LinearWarmup(optimizer: torch.optim.Optimizer, T_warmup: int) -> t.Any:
    _decay_func = functools.partial(_constant_warmup, warmup_iterations=T_warmup)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler
