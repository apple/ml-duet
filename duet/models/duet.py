#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

"""DUET implementation on top of SimCLR"""
import typing as t
import pathlib

import torch
from torch.functional import F

import tree
from torch import nn

import duet.layers as layers
from duet.models.simclr import SimCLR
from duet.augmentations import (
    ParamAugStack,
    aug_stack_from_codes,
    CODE_AUG_MAP,
)

from torchvision.utils import save_image


def get_aug_params(
    full_params: t.Dict[str, torch.Tensor],
    groups: t.List[str],
) -> t.Optional[torch.Tensor]:
    """
    Specific to DUET experiments, helper method to grab the required
    augmentation params from the full sampled ones.
    """
    if not groups:
        return None

    train_aug_names, test_aug_names, _ = aug_stack_from_codes(groups)
    color_groups = ["bri", "con", "sat", "hue"]

    def color_getter(code: str) -> torch.Tensor:
        pos = color_groups.index(code)
        return full_params["ColorJitter"][:, pos].view(-1, 1)

    def scale_getter() -> torch.Tensor:
        scale_transform = (
            "RandomResizedCrop" if "RandomResizedCrop" in full_params else "CenterCrop"
        )
        return 1 - full_params[scale_transform][:, 3].view(-1, 1)

    def param_getter(code: str) -> torch.Tensor:
        if code in color_groups:
            return color_getter(code)

        if code == "sca":
            return scale_getter()

        aug_name = CODE_AUG_MAP[code]
        return full_params[aug_name].view(-1, 1)

    params = [param_getter(code) for code in groups]

    params = torch.cat(params, dim=1)
    # assert params.amin() >= 0, f"{params.amin()} should be >= 0"
    # assert params.amax() <= 1, f"{params.amax()} should be <= 1"
    return params


DUETPoolers = {
    "max": torch.amax,
    "mean": torch.mean,
    "L1": lambda x, dim: torch.norm(x, p=1, dim=dim),
    "L2": lambda x, dim: torch.norm(x, p=2, dim=dim),
    "sum": torch.sum,
    "std": torch.std,
}


def augment_safe(x: torch.Tensor, augmentor: ParamAugStack):
    # work around FP32 issue for GaussianBlur
    original_dtype = x.dtype
    with torch.cuda.amp.autocast(enabled=False):
        xf = x.float()
        x_aug, aug_params = augmentor(xf)

        if original_dtype != x_aug.dtype:
            x_aug = x_aug.to(dtype=original_dtype)
            aug_params = tree.map_structure(
                lambda x: x.to(dtype=original_dtype), aug_params
            )  # (B, P)

    if x.is_cuda:
        aug_params = tree.map_structure(lambda t: t.cuda(), aug_params)
    return x_aug, aug_params


class DUETLayer(nn.Module):
    def __init__(
        self,
        full_dims: int,
        num_groups: int,
        group_elements: int,
        group_pooler: str = "sum",
        content_pooler: str = "mean",
        concat_pooled_reps: bool = True,
    ):
        """
        Implements a DUET layer.
        Will reshape the input N-dimensional vectors as (batch, num_groups, group_elements, dim)
        where dim is a by-product of the available input dimensionality and the group/elements chosen.

        Then a specific pooling is applied in the (num_groups, group_elements) dimensions or the dim dimension.

        :param full_dims: Full input dimensionality (eg. 2048).
        :param num_groups: Number of groups to consider (usually group refers to augmentation parameter).
        :param group_elements: Number of group elements to consider.
            This will be the "resolution" in the group dimension.
        :param group_pooler: The pooling method to use to marginalize in the group dimension.
            Applying this pooling will yield the content representation.
        :param content_pooler: The pooling method to use to marginalize in the content dimension.
            Applying this pooling will yield the group representation.
        :param concat_pooled_reps: If True, group_pooler will be applied independently per group,
            and the results will be concatenated forming the content representation.
        If False, group_pooler is applied to all groups at once.

        :returns: Content representation, group representation, extra information
        """
        super().__init__()
        if group_elements > 0:
            assert full_dims % (num_groups * group_elements) == 0
            self.feature_dim = full_dims // num_groups // group_elements
        else:
            self.feature_dim = full_dims
        self.num_params = num_groups
        self.group_elements = int(group_elements)
        self.concat_pooled_reps = concat_pooled_reps
        self.group_pooler = DUETPoolers[group_pooler]
        self.content_pooler = DUETPoolers[content_pooler]
        if self.group_elements == 0:
            print("G=0! This means we're running a baseline model.")

    def forward(
        self, z: torch.Tensor
    ) -> t.Tuple[torch.Tensor, torch.Tensor, t.Dict[str, torch.Tensor]]:
        batch_size, _ = z.shape

        if self.group_elements == 0:
            return (
                z,
                torch.zeros_like(z),
                {"std_caps_z": torch.zeros_like(z)},
            )

        zr = z.view(
            batch_size, self.num_params, self.group_elements, self.feature_dim
        )  # (B, K, G, C)

        # Pool representations in the group dimensions.
        # If concat_pooled_reps, representations are pooled per group and concatenated (yields K*C sized reps).
        # Else, representations are pooled for all groups indistinctively (yields C sized reps).
        if self.concat_pooled_reps:
            zr_c = self.group_pooler(zr, dim=2).view(batch_size, -1)  # (B, K*C)
        else:
            zr_c = self.group_pooler(zr, dim=(1, 2))  # (B, C)

        extra = {}
        return zr_c, zr, extra


class DUET(SimCLR):
    """SimCLR + DUET implementation."""

    def __init__(
        self,
        encoder_feature_size: int,
        head_output_size: int,
        head_latent_size: int,
        encoder: t.Any,
        heads: t.Dict,
        image_size_override: int,
        crop_scale: t.Tuple[float, float],
        disable_fp16_on_heads: bool = False,
        capsule_dim: int = 8,
        group_pooler: str = "sum",
        content_pooler: str = "mean",
        augs: t.List[str] = ["sca", "rot"],
        groups: t.List[str] = ["rot"],
        concat_pooled_reps: bool = True,
        **unused_kwargs,
    ):
        """DUET on a SimCLR based model.

        :param encoder_feature_size: output-size of encoding network embedding.
        :param head_output_size: output-size to use for NCE loss.
        :param head_latent_size: latent size for MLP head.
        :param encoder: pytorch encoder network
        :param heads: a dict of pytorch head networks
        :param backbone_callable_fn: a callable to modify the backbone.
        :param disable_fp16_on_heads: disables fp16 head calculations.
        :param disable_rotation: Disable rotation (becomes BYOL).
        :param capsule_dim: Number of group elements to consider. This will be the "resolution"
            in the group dimension in the capsule.
        :param group_pooler: The pooling method to use to marginalize in the group dimension.
            Applying this pooling will yield the content representation.
        :param content_pooler: The pooling method to use to marginalize in the content dimension.
            Applying this pooling will yield the group representation.
        :param concat_pooled_reps: If True, group_pooler will be applied independently per group,
            and the results will be concatenated forming the content representation.
            If False, group_pooler is applied to all groups at once.

        :returns: FoeCLR object
        :rtype: nn.Module

        """
        super().__init__(
            encoder_feature_size=encoder_feature_size,
            head_output_size=head_output_size,
            head_latent_size=head_latent_size,
            encoder=encoder,
            heads=heads,
            disable_fp16_on_heads=disable_fp16_on_heads,
        )

        assert len(groups) <= 1, f"Only 0 or 1 group supported in DUET for now. Found {groups}"

        # Augmentations are done at the minibatch level with Kornia
        self.group = groups[0] if len(groups) > 0 else None
        self.num_groups = len(groups)
        self.augs = augs
        self.augmentor = ParamAugStack(
            image_size_override=image_size_override,
            crop_scale=crop_scale,
            aug_codes=self.augs,
        )

        self.capsule_dim = capsule_dim
        if self.capsule_dim > 0:
            assert (
                encoder_feature_size % (self.num_groups * self.capsule_dim) == 0
            ), f"{encoder_feature_size} {self.num_groups} {self.capsule_dim}"
            self.feature_dim = encoder_feature_size // self.num_groups // self.capsule_dim

            # Resize head and predictor
            resized_encoder_feature_size = encoder_feature_size // self.capsule_dim
            if not concat_pooled_reps:
                resized_encoder_feature_size = resized_encoder_feature_size // self.num_groups

            self.head = nn.Sequential(
                nn.Linear(resized_encoder_feature_size, head_latent_size),
                layers.BatchNormND(head_latent_size, dim=1),
                nn.ReLU(),
                nn.Linear(head_latent_size, head_latent_size),
                layers.BatchNormND(head_latent_size, dim=1),
                nn.ReLU(),
                nn.Linear(head_latent_size, head_output_size),
                layers.BatchNormND(head_output_size, dim=1),
            )
        else:
            # Running baseline here
            print("BASELINE SimCLR in DUET CODEBASE")
            self.num_groups = 0
            resized_encoder_feature_size = encoder_feature_size

        self.foelayer = DUETLayer(
            num_groups=self.num_groups,
            full_dims=encoder_feature_size,
            group_elements=self.capsule_dim,
            group_pooler=group_pooler,
            content_pooler=content_pooler,
            concat_pooled_reps=concat_pooled_reps,
        )

    @staticmethod
    def images(predictions: t.Dict[str, torch.Tensor]) -> t.Dict[str, torch.Tensor]:
        """Extract images for plotting."""
        return {
            "reconstruction": predictions["decoded"],
            "reconstruction_target": predictions["inputs"],
        }

    def _post_process_representation(
        self,
        representation1: torch.Tensor,
        representation2: torch.Tensor,
        aug1_params: torch.Tensor,
        aug2_params: torch.Tensor,
        inputs: torch.Tensor,
        **extra_results_to_return,
    ) -> t.Dict[str, torch.Tensor]:
        """Process representation through heads."""
        with self.fp16_scope():

            def convert_fp32(reps: torch.Tensor) -> torch.Tensor:
                """Convert reprs to fp32 if we are disabling fp16 on heads."""
                return reps.float() if self.disable_fp16_on_heads else reps

            batch_size, dim = representation1.shape

            representation1 = convert_fp32(representation1)
            representation2 = convert_fp32(representation2)

            repr1_c, repr1_joint, repr1_extra = self.foelayer(representation1)
            repr2_c, repr2_joint, repr2_extra = self.foelayer(representation2)

            # infer the pooled representations through the head network
            repr1_logits = self.head(repr1_c)
            repr2_logits = self.head(repr2_c)

            # Stop-gradients to the classifier to not learn a trivially better model.
            # Always use repr1, not augmented during test.
            repr_to_classifier = (
                representation1.detach().clone().view(batch_size, -1)
            )  # (B, K*G*C) always detach!
            head_output_dict = {
                "linear_classifier": self.heads["linear_classifier"](repr_to_classifier),
            }

            return {
                "nce_logits1": repr1_logits,
                "nce_logits2": repr2_logits,
                "joint_logits1": repr1_joint,
                "joint_logits2": repr2_joint,
                "aug1_params": aug1_params,
                "aug2_params": aug2_params,
                "representations1": repr_to_classifier,
                **head_output_dict,
                **extra_results_to_return,
            }

    def forward(
        self,
        inputs: t.Dict[str, t.Union[torch.Tensor, t.Tuple[torch.Tensor]]],
        conditioning: t.Optional[torch.Tensor] = None,
        **unused_kwargs,
    ) -> t.Dict:
        unaugmented = inputs["unaugmented"]

        augmentation1, aug1_params = augment_safe(unaugmented, self.augmentor)
        # Still augment branch 2 during test, useful to compute test losses.
        # Of course, the classification loss at test will only be computed using the unaugmented branch 1.
        if not self.training:
            self.augmentor.train(True)
        augmentation2, aug2_params = augment_safe(unaugmented, self.augmentor)
        self.augmentor.train(self.training)
        # save_image(augmentation1, "aug1.png")
        # save_image(augmentation2, "aug2.png")
        # print(aug1_params)
        # print(aug2_params)
        # quit()

        # Filter aug params
        aug1_params = get_aug_params(
            aug1_params,
            groups=[
                self.group,
            ]
            if self.group
            else None,
        )
        aug2_params = get_aug_params(
            aug2_params,
            groups=[
                self.group,
            ]
            if self.group
            else None,
        )

        # Forward pass on backbone
        representation1 = self.forward_features(augmentation1)
        representation2 = self.forward_features(augmentation2)

        reconstruction_target_images = (
            torch.cat([augmentation1, augmentation2], 0) if self.training else augmentation1
        )

        return self._post_process_representation(
            representation1=representation1,
            representation2=representation2,
            aug1_params=aug1_params,
            aug2_params=aug2_params,
            inputs=reconstruction_target_images.detach().clone(),
        )
