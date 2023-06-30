#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

"""Here lie all the sub-layers."""
import typing as t
import warnings

import functools
import numpy as np
import torch
from torch import nn

import contextlib


@contextlib.contextmanager
def dummy_context():
    """Conditional with statements that do nothing."""
    yield None


class Identity(nn.Module):
    """Identity module."""

    __constants__ = ["inplace"]  # for jit-scripting

    def __init__(self, inplace: bool = False):
        """A vanilla identity jit-scriptable op."""
        super().__init__()
        self.inplace = inplace

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Does nothing."""
        if self.inplace:
            return inputs

        return inputs.clone()


class View(nn.Module):
    """View as an module."""

    __constants__ = ["shape"]  # for jit-scripting

    def __init__(self, shape):
        """The target shape for the view."""
        super().__init__()
        self.shape = shape

    def __repr__(self):
        """Extra str info."""
        return "View({})".format(self.shape)

    def forward(self, inputs):
        """A new view on inputs based on self.shape."""
        return inputs.contiguous().view(*self.shape)


class Squeezer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze()


class BatchNormND(nn.modules.batchnorm._BatchNorm):  # noqa
    """Unified BN framework for all dimensions."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dim: int = 2,
    ):
        """Generic batchnorm implementation.

        :param num_features: number of features
        :param eps: numerical stability
        :param momentum: for BN EMA (NOTE: is inverse of what you think).
        :param affine: adds learnable beta and gamma affine terms.
        :param track_running_stats: tracks
        :param dim: dimension of data to work with.
        :returns:

        """
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.dim = dim
        self.warn_once = False

        self.check_fn = self._check_any
        dim_mapper = {
            1: BatchNormND._check_1d,
            2: BatchNormND._check_2d,
            3: BatchNormND._check_3d,
        }
        if self.dim in dim_mapper:
            self.check_fn = dim_mapper[self.dim]

    def extra_repr(self):
        """Adds dim to regular BN repr."""
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}, dim={dim}".format(**self.__dict__)
        )

    @staticmethod
    def _copy_buffers_and_params(source, dest):
        """Copies source BN params and buffers into dest."""
        if source.affine:
            with torch.no_grad():
                dest.weight = source.weight
                dest.bias = source.bias

        dest.running_mean = source.running_mean
        dest.running_var = source.running_var
        dest.num_batches_tracked = source.num_batches_tracked

        if hasattr(source, "qconfig"):
            dest.qconfig = source.qconfig

        return dest

    @classmethod
    def convert_to_batchnorm_nd(cls, module):
        """Helper to convert all nn.BatchNorm*D layers to BatchNormND while preserving dim."""
        module_output = module
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            dim_mapper = {
                nn.BatchNorm1d: 1,
                nn.BatchNorm2d: 2,
                nn.BatchNorm3d: 3,
            }

            module_output = BatchNormND(
                num_features=module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
                dim=dim_mapper[type(module)],
            )
            module_output = BatchNormND._copy_buffers_and_params(source=module, dest=module_output)

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_to_batchnorm_nd(child))

        del module
        return module_output

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        """Helper to convert all BatchNormND layers to SyncBN while preserving dim."""
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert hasattr(
                module, "dim"
            ), "Only BatchNormND supported for invertible SyncBN conversion."

            module_output = torch.nn.SyncBatchNorm(
                num_features=module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
                process_group=process_group,
            )
            module_output.dim = module.dim
            module_output = BatchNormND._copy_buffers_and_params(source=module, dest=module_output)

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))

        del module
        return module_output

    @classmethod
    def revert_sync_batchnorm(cls, module):
        """Helper to convert all SyncBatchNorm layers to BatchNormND."""
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
            assert hasattr(module, "dim"), "Only BatchNormND converted SyncBN can be inverted."

            module_output = BatchNormND(
                num_features=module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
                dim=module.dim,
            )
            module_output = BatchNormND._copy_buffers_and_params(source=module, dest=module_output)

        for name, child in module.named_children():
            module_output.add_module(name, cls.revert_sync_batchnorm(child))

        del module
        return module_output

    @staticmethod
    def _check_1d(input_dim: int) -> None:
        """Check sizing for a 1d input (eg: features or time+features)."""
        if input_dim not in (2, 3):
            raise ValueError("BN expected 2D or 3D input (got {}D input)".format(input_dim))

    @staticmethod
    def _check_2d(input_dim: int) -> None:
        """Check sizing for a 2d input (eg: images)."""
        if input_dim != 4:
            raise ValueError("BN expected 4D input (got {}D input)".format(input_dim))

    @staticmethod
    def _check_3d(input_dim: int) -> None:
        """Check sizing for a 3d input (eg: video)."""
        if input_dim != 5:
            raise ValueError("BN expected 5D input (got {}D input)".format(input_dim))

    def _check_any(self, input_dim: int) -> None:
        """Generic warning function for unknown dimensions."""
        if not self.warn_once:
            warnings.warn(f"BN check_input_dim not implemented for {input_dim}D input")
            self.warn_once = True

    def _check_input_dim(self, input: torch.Tensor):
        """Checks input dimensions (kept as input for override)."""
        input_dim = input.dim()
        self.check_fn(input_dim)
