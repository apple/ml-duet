#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

"""Downstream task objectives."""

import functools
import typing as t

import torch
import torch.nn.functional as F

from .common import LOSS_TO_OPTIMIZE


def correct_matrix_to_topk(correct: torch.Tensor, ks: t.Iterable[int] = (1,)) -> t.List[float]:
    """The aggregation comparison piece of topk."""
    batch_size = correct.shape[1]
    res = []
    for k in ks:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def topk(output: torch.Tensor, target: torch.Tensor, ks: t.Iterable[int] = (1,)):
    """Computes the accuracy over the k top predictions for the specified values of k.

    If requested k is larger than the number of classes, we take the value of k as the
    number of classes.

    From: https://bit.ly/2Y1MOAq
    """
    num_classes = output.shape[-1]
    ks = tuple(min(k, num_classes) for k in ks)
    with torch.no_grad():
        maxk = max(ks)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        return correct_matrix_to_topk(correct, ks=ks)


def _calc_topks_dict(
    logits: torch.Tensor, labels: torch.Tensor, top_ks: t.Iterable[int] = (1, 5)
) -> t.Dict[str, torch.Tensor]:
    """Calculate the top-k and upack to a nice dict."""
    accuracies = topk(output=logits, target=labels, ks=top_ks)
    return {f"top_{k}_mean": accuracies[idx] for idx, k in enumerate(top_ks)}


_REDUCTION_MAP = {  # helper to grab reduction fn
    "mean": torch.mean,
    "sum": torch.sum,
    "none": lambda tensor: tensor,
}


def smooth_one_hot(
    true_labels: torch.Tensor, num_classes: int, smoothing: float = 0.0
) -> torch.Tensor:
    """Smooth one-hot vector creation.

    From: https://github.com/pytorch/pytorch/issues/7455

    If smoothing == 0 --> one-hot.
    If 0 < smoothing < 1 --> soft tensor.

    :param true_labels: true label tensor.
    :param num_classes: number of classes.
    :param smoothing: smoothing co-eff.
    :returns: tensor

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing

    with torch.no_grad():
        true_dist = torch.empty(size=(true_labels.size(0), num_classes), device=true_labels.device)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)

    return true_dist


def _soft_target_softmax_cross_entropy(
    input: torch.Tensor,  # kept for kwarg compat.
    target: torch.Tensor,
    reduction: str = "mean",
    dim: int = -1,
):
    """Targets are already vectorized in some form."""
    loss = -target * F.log_softmax(input, dim=dim)
    return _REDUCTION_MAP[reduction](torch.sum(loss, dim=dim))


def cross_entropy_with_label_smoothing(
    input: torch.Tensor,  # kept for kwarg compat.
    target: torch.Tensor,
    smoothing: float,
    reduction: str = "mean",
    dim: int = -1,
):
    """Compatibility wrapper with F.cross_entropy with label smoothing support."""
    num_classes = input.shape[dim]
    smooth_labels = smooth_one_hot(true_labels=target, num_classes=num_classes, smoothing=smoothing)
    return soft_target_softmax_cross_entropy(
        input=input, target=smooth_labels, reduction=reduction, dim=dim
    )


def softmax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    label_smoothing: float = 0.0,
    elementwise: bool = False,
    **unused_kwargs,
) -> t.Dict[str, torch.Tensor]:
    """Computes softmax cross entropy (elementwise if requested) and metrics."""
    top_ks_dict = _calc_topks_dict(logits, labels)

    # either do label smoothing or standard XE
    loss_fn = (
        functools.partial(
            cross_entropy_with_label_smoothing,
            smoothing=label_smoothing,
        )
        if label_smoothing > 0
        else F.cross_entropy
    )

    if elementwise:
        loss = loss_fn(input=logits, target=labels, reduction="none")

        # Handle 2 augmentations
        return {
            "loss": loss.reshape(-1, batch_size).mean(0),
            LOSS_TO_OPTIMIZE: loss.mean(),
            **top_ks_dict,
        }

    return {
        LOSS_TO_OPTIMIZE: loss_fn(input=logits, target=labels, reduction="mean"),
        **top_ks_dict,
    }


def soft_target_softmax_cross_entropy(
    logits: torch.Tensor,  # [B, F]
    labels: torch.Tensor,  # now [B, F]
    batch_size: int,
    elementwise: bool = False,
    **unused_kwargs,
) -> t.Dict[str, torch.Tensor]:
    """Softmax-cross entropy with soft targets (already vectorized)."""
    if labels.dim() == 1:  # eg: mixup not applied at test
        num_classes = logits.shape[-1]
        labels = F.one_hot(labels, num_classes=num_classes)

    top_ks_dict = _calc_topks_dict(logits=logits, labels=labels.argmax(-1))
    loss_fn = _soft_target_softmax_cross_entropy
    if elementwise:
        loss = loss_fn(input=logits, target=labels, reduction="none")

        # Handle 2 augmentations
        return {
            "loss": loss.reshape(-1, batch_size).mean(0),
            LOSS_TO_OPTIMIZE: loss.mean(),
            **top_ks_dict,
        }

    return {
        LOSS_TO_OPTIMIZE: loss_fn(input=logits, target=labels, reduction="mean"),
        **top_ks_dict,
    }


def binary_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    elementwise: bool = False,
    **unused_kwargs,
) -> t.Dict[str, torch.Tensor]:
    """Compute BCE loss (elementwise if requested)."""
    # TODO(jramapuram): add bce metrics
    if elementwise:
        loss = F.binary_cross_entropy_with_logits(
            input=logits, target=labels.float(), reduction="none"
        )

        # Handle 2 augmentations
        return {
            "loss": loss.reshape(-1, batch_size, loss.shape[-1]).mean(0),
            LOSS_TO_OPTIMIZE: loss.mean(),
        }

    return {
        LOSS_TO_OPTIMIZE: F.binary_cross_entropy_with_logits(
            input=logits, target=labels.float(), reduction="mean"
        )
    }


def get_classifier_loss(loss_type: str):
    """Return the correct loss function."""
    classifier_loss_types = {
        "softmax_cross_entropy": softmax_cross_entropy,
        "soft_target_softmax_cross_entropy": soft_target_softmax_cross_entropy,
        "binary_cross_entropy": binary_cross_entropy,
    }
    assert loss_type in classifier_loss_types, f"{loss_type} loss not defined."
    return classifier_loss_types[loss_type]
