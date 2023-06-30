#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

"""MSE Loss."""

import typing as t

import torch
import torch.nn.functional as F

from .common import LOSS_TO_OPTIMIZE


def mse_loss(
    x: torch.Tensor, y: torch.Tensor, elementwise: bool = False
) -> t.Dict[str, torch.Tensor]:
    """Mean-squared error loss."""
    loss_mean = F.mse_loss(x, y)
    if elementwise:
        batch_size = x.shape[0]
        loss = torch.sum(
            F.mse_loss(x.view([batch_size, -1]), y.view([batch_size, -1]), reduction="none"),
            dim=-1,
        )
        return {"loss": loss, LOSS_TO_OPTIMIZE: loss_mean}

    return {LOSS_TO_OPTIMIZE: loss_mean}
