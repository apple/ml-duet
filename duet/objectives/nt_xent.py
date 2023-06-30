#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

"""NT-xent from SimCLR."""

import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LOSS_TO_OPTIMIZE
from .classification import topk


def lp_normalize(
    tensor: torch.Tensor,
    p: float = 2.0,
    dim: t.Optional[int] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize a tensor over dim using the Lp-norm."""
    assert p > 0
    pow_sum = torch.sum(torch.pow(tensor, exponent=p), dim=dim, keepdim=True)
    pow_sum = torch.max(pow_sum, torch.ones_like(pow_sum) * eps)
    inv_norm = torch.pow(pow_sum, exponent=-1 / p)
    return tensor * inv_norm


def l2_normalize(
    tensor: torch.Tensor, dim: t.Optional[int] = None, eps: float = 1e-12
) -> torch.Tensor:
    """Normalize a tensor over dim using the L2-norm."""
    return lp_normalize(tensor=tensor, p=2, dim=dim, eps=eps)


def nt_xent(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    temperature: float = 0.1,
    elementwise: bool = False,
) -> t.Dict[str, torch.Tensor]:
    """NT-XENT Loss from SimCLR.

    :param embedding1: embedding of augmentation1
    :param embedding2: embedding of augmentation2
    :param temperature: nt-xent temperature, usually 0.1.
    :param num_replicas: number of compute devices
    :param elementwise: If true returns elementwise nce loss in addition to the aggregated lose (default=False).
    :returns: scalar loss and optionally elementwise loss
    :rtype: float32

    """
    batch_size = embedding1.shape[0]
    # feature_size = embedding1.shape[-1]
    # num_replicas = dist.get_world_size() if dist.is_initialized() else 1
    LARGE_NUM = 1e9

    # normalize both embeddings
    embedding1 = l2_normalize(embedding1, dim=-1)
    embedding2 = l2_normalize(embedding2, dim=-1)

    # Prepared to work distributed, but not implemented yet.
    embedding1_full = embedding1
    embedding2_full = embedding2
    masks = F.one_hot(torch.arange(batch_size, device=embedding1.device), batch_size)
    labels = torch.arange(batch_size, device=embedding1.device).type(torch.int64)

    # Matmul-to-mask
    logits_aa = torch.matmul(embedding1, embedding1_full.T) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(embedding2, embedding2_full.T) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(embedding1, embedding2_full.T) / temperature
    logits_ba = torch.matmul(embedding2, embedding1_full.T) / temperature

    # Use our standard cross-entropy loss which uses log-softmax internally.
    # Concat on the feature dimension to provide all features for standard softmax-xent
    logits_abaa = torch.cat([logits_ab, logits_aa], 1)
    loss_a = F.cross_entropy(
        input=logits_abaa,
        target=labels,
        reduction="none",
    )
    logits_babb = torch.cat([logits_ba, logits_bb], 1)
    loss_b = F.cross_entropy(
        input=logits_babb,
        target=labels,
        reduction="none",
    )

    loss = loss_a + loss_b
    loss_mean = torch.mean(loss)

    # compute top-1 and top-5 for InfoNCE.
    top_ks = (1, 5) if logits_abaa.shape[0] > 5 else (1,)
    accuracies_loss_a = topk(output=logits_abaa, target=labels, ks=top_ks)
    accuracies_loss_b = topk(output=logits_babb, target=labels, ks=top_ks)

    # aggregate the metrics.
    perf_dict = {}
    for idx, k in enumerate(top_ks):
        perf_dict[f"ntxent_ABAA_top{k}_mean"] = accuracies_loss_a[idx]
        perf_dict[f"ntxent_BABB_top{k}_mean"] = accuracies_loss_b[idx]
        perf_dict[f"ntxent_total_top{k}_mean"] = (1.0 / len(top_ks)) * sum(
            accuracies_loss_a[idx] + accuracies_loss_b[idx]
        )

    if elementwise:
        return {"loss": loss, LOSS_TO_OPTIMIZE: loss_mean, **perf_dict}

    return {LOSS_TO_OPTIMIZE: loss_mean, **perf_dict}


class NTXent(nn.Module):
    """NT xent loss module from SimCLR."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        elementwise: bool = False,
    ) -> t.Dict[str, torch.Tensor]:
        """NT-XENT Loss from SimCLR.

        :param embedding1: embedding of augmentation1
        :param embedding2: embedding of augmentation2
        :param num_replicas: number of compute devices
        :param elementwise: If true returns elementwise nce loss in addition to the aggregated lose (default=False).
        :returns: scalar loss and optionally elementwise loss
        :rtype: float32

        """
        return nt_xent(
            embedding1=embedding1,
            embedding2=embedding2,
            temperature=self.temperature,
            elementwise=elementwise,
        )
