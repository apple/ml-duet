#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

"""ESSL loss."""

import typing as t

import torch

from .common import LOSS_TO_OPTIMIZE
from .nt_xent import nt_xent
from .classification import softmax_cross_entropy
from .autoencoding import mse_loss


def essl_loss(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    essl_logits: torch.Tensor,
    essl_labels: torch.Tensor,
    lambd: float,
    groups: t.List[str] = ["ron"],
    temperature: float = 0.1,
    elementwise: bool = False,
) -> t.Dict[str, torch.Tensor]:
    """
    Implements NtXENT + FOE loss

    :param embedding1: Content representation from branch 1.
    :param embedding2: Content representation from branch 2.
    :param joint_logits1: Full (B, K, G, C) representation from branch 1.
    :param joint_logits2: Full (B, K, G, C) representation from branch 2.
    :param aug1_params: Augmentation parameters applied to branch 1.
    :param aug2_params: Augmentation parameters applied to branch 2.
    :param lambd: Lambda parameter weighing the equivariance loss.
    :param groups: Group codes to learn equivariance to.
    :param temperature: Nt Xent temperature.
    :param elementwise: Return elementwise loss.
    :param training: True if training mode.

    :return: A dict with LOSS_TO_OPTIMIZE + extra losses for debugging.
    """

    # Compute normal SimCLR nt_xent loss with content representations.
    nce_loss_dict = nt_xent(
        embedding1=embedding1,
        embedding2=embedding2,
        temperature=temperature,
        elementwise=elementwise,
    )

    # Compute ESSL loss hereafter.
    assert len(groups) == 1
    use_mse = groups[0] in ["rot", "hue", "sat", "bri", "con"]
    if use_mse:
        essl_loss_dict = mse_loss(
            x=essl_logits,
            y=essl_labels,
        )
    else:
        essl_loss_dict = softmax_cross_entropy(
            logits=essl_logits,
            labels=essl_labels,
            batch_size=essl_logits.shape[0],
        )

    main_loss = nce_loss_dict[LOSS_TO_OPTIMIZE] + lambd * essl_loss_dict[LOSS_TO_OPTIMIZE]
    essl_loss_dict = {"essl_" + k: v for k, v in essl_loss_dict.items()}

    return {
        LOSS_TO_OPTIMIZE: main_loss,
        "nce_loss_mean": nce_loss_dict[LOSS_TO_OPTIMIZE],
        **essl_loss_dict,
    }
