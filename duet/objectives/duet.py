#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

"""DUET loss."""

import functools
import typing as t

import torch
from torch.functional import F

from .common import LOSS_TO_OPTIMIZE
from .nt_xent import nt_xent


def jensen_shannon_div(p: torch.tensor, q: torch.tensor, reduction: str = "batchmean"):
    """
    Expects p and q softmax'd.
    """
    kl = torch.nn.KLDivLoss(reduction=reduction, log_target=True)
    p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
    m = (0.5 * (p + q)).log()
    return 0.5 * (kl(m, p.log()) + kl(m, q.log()))


def js_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    softmax_logits: bool = True,
    softmax_targets: bool = False,
) -> torch.Tensor:
    return jensen_shannon_div(
        F.softmax(logits, dim=-1) if softmax_logits else logits,
        F.softmax(targets, dim=-1) if softmax_targets else targets,
        reduction="batchmean",
    )


def make_gaussian_targets(
    params: torch.Tensor, size: int, std: float = 0.2, normalize: str = "sum"
) -> torch.Tensor:
    """
    Params is of shape (B, K), expected all normalized between 0 and 1.
    Return a tensor of shape (B, K, size) where each element (B, K) is a
    discretized gaussian centered at param.
    """
    assert normalize in [
        "none",
        "sum",
    ]
    b, k = params.shape
    std_all = std * torch.ones_like(params)
    normal = torch.distributions.normal.Normal(loc=params, scale=std_all)
    # normal expects an input of size (B, K, num_samples)
    # where num_samples for us is a linear interpolation between 0 and 1, with size samples.
    partition_divs = 100
    fine_grid = size * partition_divs
    x = torch.linspace(0.0, 1.0, fine_grid, device=params.device).repeat(b, k, 1).permute(2, 0, 1)
    # Evaluate gaussian
    y = normal.log_prob(x).exp()
    y = y.permute(1, 2, 0)

    # Integrate in bounds of partition
    samples = y.view(b, k, size, partition_divs).sum(-1) * 1.0 / size / partition_divs

    if normalize == "sum":
        samples = samples / torch.sum(samples, dim=-1, keepdim=True)
    return samples


def make_vonmises_targets(
    params: torch.Tensor, size: int, std: float = 0.5, normalize: str = "sum"
) -> torch.Tensor:
    """
    Params is of shape (B, K), expected all normalized between 0 and 1.
    Return a tensor of shape (B, K, size) where each element (B, K) is a
    discretized Von Mises distribution centered at param.
    """
    assert normalize in [
        "none",
        "sum",
    ]
    b, k = params.shape

    pi2 = 2 * torch.pi
    std_all = std * torch.ones_like(params)
    normal = torch.distributions.VonMises(loc=params * pi2, concentration=std_all)
    # normal expects an input of size (B, K, num_samples)
    # where num_samples for us is a linear interpolation between 0 and 1, with size samples.
    partition_divs = 100  # fine-grained interval for integration
    fine_grid = size * partition_divs
    x = torch.linspace(0.0, pi2, fine_grid, device=params.device).repeat(b, k, 1).permute(2, 0, 1)
    # Evaluate gaussian
    y = normal.log_prob(x).exp()
    y = y.permute(1, 2, 0)

    # Integrate in bounds of partition
    samples = y.view(b, k, size, partition_divs).sum(-1) * pi2 / size / partition_divs

    if normalize == "sum":
        samples = samples / torch.sum(samples, dim=-1, keepdim=True)
    return samples


def make_targets(
    params: torch.Tensor,
    size: int,
    std: float = 0.2,
    normalize: str = "sum",
    target: str = "ga",
) -> torch.Tensor:
    if target == "vm":
        fn = make_vonmises_targets
    else:
        fn = make_gaussian_targets

    return fn(params=params, size=size, std=std, normalize=normalize)


def nt_xent_duet(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    joint_logits1: torch.Tensor,
    joint_logits2: torch.Tensor,
    aug1_params: torch.Tensor,
    aug2_params: torch.Tensor,
    lambd: float,
    capsule_dim: int,
    sigma: float,
    groups: t.List[str],
    target: str,
    temperature: float = 0.1,
    elementwise: bool = False,
    training: bool = True,
) -> t.Dict[str, torch.Tensor]:
    """
    Implements NtXENT + DUET loss

    :param embedding1: Content representation from branch 1.
    :param embedding2: Content representation from branch 2.
    :param joint_logits1: Full (B, K, G, C) representation from branch 1.
    :param joint_logits2: Full (B, K, G, C) representation from branch 2.
    :param aug1_params: Augmentation parameters applied to branch 1.
    :param aug2_params: Augmentation parameters applied to branch 2.
    :param lambd: Lambda parameter weighing the group loss.
    :param sigma: Sigma for gaussian targets.
    :param group: Group code to gain equivariance to.
    :param target: Target distribution, either `ga` (Gaussian) or `vm` (von-Mises).
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

    # Just compute nt_xent if baseline case.
    if capsule_dim == 0:
        return nce_loss_dict

    # Compute group loss hereafter.
    batch_size, num_groups, capsule_dim, feature_dim = joint_logits1.shape

    assert num_groups == 1, f"Number of groups larger than 1 not implemented."
    assert len(groups) == 1, f"Number of groups larger than 1 not implemented."

    embedding1_g = torch.mean(joint_logits1, dim=-1)
    embedding2_g = torch.mean(joint_logits2, dim=-1)

    make_targets_fn = functools.partial(
        make_targets,
        size=capsule_dim,
        std=sigma,
        target=target,
    )

    g1_targets = make_targets_fn(aug1_params).clone().detach()  # (B, K, G)
    g2_targets = make_targets_fn(aug2_params).clone().detach()  # (B, K, G)

    def per_group_loss(n_groups):
        gi_view2 = embedding2_g.chunk(n_groups, dim=1)
        ti_view2 = g2_targets.chunk(n_groups, dim=1)
        if training:
            gi_view1 = embedding1_g.chunk(n_groups, dim=1)
            ti_view1 = g1_targets.chunk(n_groups, dim=1)
            logits_gi = [torch.cat([gi_view1[i], gi_view2[i]], dim=1) for i in range(n_groups)]
            targets_gi = [torch.cat([ti_view1[i], ti_view2[i]], dim=1) for i in range(n_groups)]
        else:
            # If test, only use view2 (augmented)
            logits_gi = gi_view2
            targets_gi = ti_view2

        return logits_gi, targets_gi

    logits_gi, targets_gi = per_group_loss(num_groups)
    loss_jsd_gi = [js_loss(l, t) for l, t in zip(logits_gi, targets_gi)]

    # Store group losses
    dict_loss_jsd_gi = {f"group_{i}_loss_mean": loss_jsd_gi[i] for i in range(num_groups)}

    # # Use all groups jointly for optimization
    # logits_gi, targets_gi = per_group_loss(n_groups=1)
    # loss_jsd_gi = [g_loss(l, t) for l, t in zip(logits_gi, targets_gi)]
    # loss_jsd_g = sum(loss_jsd_gi) / len(loss_jsd_gi)

    # Final group loss is the weighted mean of the per-group losses
    loss_jsd_g = lambd * sum(loss_jsd_gi) / len(loss_jsd_gi)

    # if len(lambd) == 1:
    #     loss_jsd_g = lambd[0] * sum(loss_jsd_gi) / len(loss_jsd_gi)
    # else:
    #     assert len(lambd) == len(loss_jsd_gi)
    #     loss_jsd_gi = [b * l for b, l in zip(lambd, loss_jsd_gi)]
    #     loss_jsd_g = sum(loss_jsd_gi) / len(loss_jsd_gi)

    main_loss = nce_loss_dict[LOSS_TO_OPTIMIZE] + loss_jsd_g

    return {
        LOSS_TO_OPTIMIZE: main_loss,
        "nce_loss_mean": nce_loss_dict[LOSS_TO_OPTIMIZE],
        "group_loss_mean": loss_jsd_g,
        **dict_loss_jsd_gi,
    }
