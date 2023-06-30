#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

"""ESSL implementation on top of SimCLR"""

import functools
import typing as t
import pathlib

import torch
from torch.functional import F

from torch import nn
import kornia as K

import duet.layers as layers
from duet.models.simclr import SimCLR
from duet.augmentations import (
    ParamAugStack,
)
from duet.models.duet import augment_safe


def rotate_images(
    images: torch.Tensor, multiple: bool = True
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate 4-fold.

    From: https://github.com/rdangovs/essl/blob/70b2cd0b145308c857ed8ebf62b89a1a0eebf765/cifar10/main.py#L51-L70
    """

    nimages = images.shape[0]
    multiplicity = 4 if multiple else 1
    n_rot_images = multiplicity * nimages

    # rotate images all 4 ways at once
    rotated_images = torch.zeros(
        [n_rot_images, images.shape[1], images.shape[2], images.shape[3]]
    ).to(images.device)
    rot_classes = torch.zeros([n_rot_images]).long().cuda()

    nsplit = nimages if multiplicity else nimages // 4
    if multiplicity == 1:
        in_images = [
            images[:nsplit],
            images[nsplit : 2 * nsplit],
            images[2 * nsplit : 3 * nsplit],
            images[3 * nsplit :],
        ]
    elif multiplicity == 4:
        in_images = [images, images, images, images]

    assert n_rot_images == 4 * nsplit

    rotated_images[:nsplit] = in_images[0]
    # rotate 90
    rotated_images[nsplit : 2 * nsplit] = in_images[1].flip(3).transpose(2, 3)
    rot_classes[nsplit : 2 * nsplit] = 1
    # rotate 180
    rotated_images[2 * nsplit : 3 * nsplit] = in_images[2].flip(3).flip(2)
    rot_classes[2 * nsplit : 3 * nsplit] = 2
    # rotate 270
    rotated_images[3 * nsplit : 4 * nsplit] = in_images[3].transpose(2, 3).flip(3)
    rot_classes[3 * nsplit : 4 * nsplit] = 3

    return rotated_images, rot_classes


def rotate_cont_images(
    images: torch.Tensor, multiple: bool = True
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    tx = K.augmentation.RandomRotation(degrees=90, p=1.0)
    tx_images = tx(images)
    tx_classes = tx._params["degrees"] / 180.0
    return tx_images, tx_classes.view(-1, 1)


def jitter_images(
    images: torch.Tensor,
    tx_conf: t.Dict[str, float],
    multiple: bool = True,
    normalize: bool = False,
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    assert len(tx_conf) == 1
    tx_name = list(tx_conf.keys())[0]
    tx_value = tx_conf[tx_name]
    tx = K.augmentation.ColorJitter(**tx_conf, p=1.0)
    tx_images = tx(images)
    tx_classes = tx._params[f"{tx_name}_factor"]
    if normalize:
        if tx_name == "hue":
            minv, maxv = -tx_value, tx_value
        else:
            minv, maxv = 1 - tx_value, 1 + tx_value
        tx_classes = (tx_classes - minv) / (maxv - minv)
    return tx_images, tx_classes.view(-1, 1)


def hflip_images(
    images: torch.Tensor, multiple: bool = True
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    nimages = images.shape[0]
    multiplicity = 2 if multiple else 1
    n_tx_images = multiplicity * nimages

    nsplit = nimages if multiplicity else nimages // 2
    if multiplicity == 1:
        in_images = [images[:nsplit], images[nsplit:]]
    elif multiplicity == 2:
        in_images = [images, images]

    tx_images = torch.zeros([n_tx_images, images.shape[1], images.shape[2], images.shape[3]]).to(
        images.device
    )
    tx_classes = torch.zeros([n_tx_images]).long().to(images.device)
    tx_images[:nsplit] = in_images[0]
    tx_images[nsplit:] = in_images[1].flip(3)  # along W
    tx_classes[nsplit:] = 1
    return tx_images, tx_classes


def vflip_images(
    images: torch.Tensor, multiple: bool = True
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    nimages = images.shape[0]
    multiplicity = 2 if multiple else 1
    n_tx_images = multiplicity * nimages

    nsplit = nimages if multiplicity else nimages // 2
    if multiplicity == 1:
        nsplit = nimages // 2
        in_images = [images[:nsplit], images[nsplit:]]
    elif multiplicity == 2:
        nsplit = nimages
        in_images = [images, images]

    tx_images = torch.zeros([n_tx_images, images.shape[1], images.shape[2], images.shape[3]]).to(
        images.device
    )
    tx_classes = torch.zeros([n_tx_images]).long().to(images.device)
    tx_images[:nsplit] = in_images[0]
    tx_images[nsplit:] = in_images[1].flip(2)  # along H
    tx_classes[nsplit:] = 1
    return tx_images, tx_classes


def grayscale_images(
    images: torch.Tensor, multiple: bool = True
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    nimages = images.shape[0]
    multiplicity = 2 if multiple else 1
    n_tx_images = multiplicity * nimages
    tx_images = torch.zeros([n_tx_images, images.shape[1], images.shape[2], images.shape[3]]).to(
        images.device
    )
    tx_classes = torch.zeros([n_tx_images]).long().to(images.device)

    if multiplicity == 1:
        nsplit = nimages // 2
        in_images = [images[:nsplit], images[nsplit:]]
    elif multiplicity == 2:
        nsplit = nimages
        in_images = [images, images]

    tx_images[:nsplit] = in_images[0]
    tx_images[nsplit:] = K.color.gray.rgb_to_grayscale(in_images[1]).repeat(1, 3, 1, 1)
    tx_classes[nsplit:] = 1
    return tx_images, tx_classes


class ESSL(SimCLR):
    """SimCLR + ESSL implementation."""

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
        augs: t.List[str] = ["sca"],
        groups: t.List[str] = ["ron"],
        lambd: t.List[float] = 0.4,
        **unused_kwargs,
    ):
        """FOE on a SimCLR based model.

        :param encoder_feature_size: output-size of encoding network embedding.
        :param head_output_size: output-size to use for NCE loss.
        :param head_latent_size: latent size for MLP head.
        :param encoder: pytorch encoder network
        :param heads: a dict of pytorch head networks
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

        assert len(groups) == 1, f"Only 1 group supported in ESSL for now. Found {groups}"

        self.group = groups[0]
        self.lambda_loss = lambd
        # ESSL considers 2 cases:
        #   insensitive to aug: SimCLR baseline with aug added to the stack.
        #   sensitive to aug: SimCLR without aug, but aug on ESSL head.
        # This boils down to removing "ron" from ESSL augs.
        # TODO: Should we remove "ron" from clr_augs if lambda > 0?
        clr_augs = augs.copy()
        essl_augs = augs.copy()
        # # Remove rotation from ESSL augs if we need to predict it (we'll add it after through rotate_images().
        # try:
        #     essl_augs.remove("ron")
        # except KeyError:
        #     pass

        # Augmentations are done at the minibatch level with Kornia
        self.augmentor = ParamAugStack(
            image_size_override=image_size_override,
            crop_scale=crop_scale,
            aug_codes=clr_augs,
        )

        # Create a different augmentor that crops to half the size
        # From: https://github.com/rdangovs/essl/blob/70b2cd0b145308c857ed8ebf62b89a1a0eebf765/cifar10/main.py#L26-L37
        self.augmentor_essl = ParamAugStack(
            image_size_override=image_size_override // 2,
            crop_scale=crop_scale,
            aug_codes=essl_augs,
        )

        # ESSL predictor head
        # From https://github.com/rdangovs/essl/blob/70b2cd0b145308c857ed8ebf62b89a1a0eebf765/cifar10/main.py#L146-L153
        out_map = {
            "ron": 4,
            "fli": 2,
            "flv": 2,
            "gra": 2,
            "rot": 1,
            "bri": 1,
            "con": 1,
            "sat": 1,
            "hue": 1,
        }
        self.predictor_essl = nn.Sequential(
            nn.Linear(encoder_feature_size, head_latent_size),
            nn.LayerNorm(head_latent_size),
            nn.ReLU(),  # first layer
            nn.Linear(head_latent_size, head_latent_size),
            nn.LayerNorm(head_latent_size),
            nn.ReLU(),
            nn.Linear(head_latent_size, out_map[self.group]),  # output layer
        )

        multiple = True

        tx_map = {
            "ron": functools.partial(rotate_images, multiple=multiple),
            "fli": functools.partial(hflip_images, multiple=multiple),
            "flv": functools.partial(vflip_images, multiple=multiple),
            "gra": functools.partial(grayscale_images, multiple=multiple),
            "rot": rotate_cont_images,
            "bri": functools.partial(jitter_images, tx_conf={"brightness": 0.4}, normalize=False),
            "con": functools.partial(jitter_images, tx_conf={"contrast": 0.4}, normalize=False),
            "sat": functools.partial(jitter_images, tx_conf={"saturation": 0.4}, normalize=False),
            "hue": functools.partial(jitter_images, tx_conf={"hue": 0.1}, normalize=False),
        }
        self.equi_transform = tx_map[self.group]

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
        representation_essl: torch.Tensor,
        essl_labels: torch.Tensor,
        inputs: torch.Tensor,
        **extra_results_to_return,
    ) -> t.Dict[str, torch.Tensor]:
        """Process representation through heads."""
        with self.fp16_scope():

            def convert_fp32(reps: torch.Tensor) -> torch.Tensor:
                """Convert reprs to fp32 if we are disabling fp16 on heads."""
                return reps.float() if self.disable_fp16_on_heads else reps

            representation1 = convert_fp32(representation1)
            representation2 = convert_fp32(representation2)
            representation_essl = convert_fp32(representation_essl)

            batch_size, dim = representation1.shape

            # infer the pooled representations through the head network
            repr1_logits = self.head(representation1)
            repr2_logits = self.head(representation2)
            essl_logits = self.predictor_essl(representation_essl)

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
                "essl_logits": essl_logits,
                "essl_labels": essl_labels,
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
        # Augment using the reduced in size augmentor
        augmentation_essl, aug_essl_params = augment_safe(unaugmented, self.augmentor_essl)
        representation1 = self.forward_features(augmentation1)
        representation2 = self.forward_features(augmentation2)

        # Also push the rotated images through backbone. We rotate the images coming from the reduced size augmentor.
        # TODO: Improve this, now we need a hack.
        # 1. unnormalize imgs
        augmentation_essl = augmentation_essl + 0.5
        # 2. apply transform
        rotated_images, rotated_labels = self.equi_transform(augmentation_essl)
        # 3. normalize again
        rotated_images = rotated_images - 0.5
        rotated_labels = rotated_labels.to(rotated_images.device)
        if self.lambda_loss <= 0.0:
            with torch.no_grad():
                representation_essl = self.forward_features(rotated_images)
        else:
            representation_essl = self.forward_features(rotated_images)

        reconstruction_target_images = (
            torch.cat([augmentation1, augmentation2], 0) if self.training else augmentation1
        )

        return self._post_process_representation(
            representation1=representation1,
            representation2=representation2,
            representation_essl=representation_essl,
            essl_labels=rotated_labels,
            inputs=reconstruction_target_images.detach().clone(),
        )
