#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

"""SimCLR implementation."""
import functools
import typing as t

import torch
from torch import nn

import duet.layers as layers

from duet.models.encoders.cifar10_resnet import Cifar10_ResNet


class SimCLR(nn.Module):
    """Simple SIMCLR implementation."""

    def __init__(
        self,
        encoder_feature_size: int,
        head_output_size: int,
        head_latent_size: int,
        encoder: t.Any,
        heads: t.Dict,
        disable_fp16_on_heads: bool = False,
        **unused_kwargs,
    ):
        """SimCLR model.

        :param encoder_feature_size: output-size of base network embedding.
        :param head_output_size: output-size to use for NCE loss.
        :param head_latent_size: latent size for MLP head.
        :param encoder: pytorch backbone network
        :param heads: a dict of pytorch head networks
        :param disable_fp16_on_heads: disables fp16 head calculations.
        :returns: SimCLR object
        :rtype: nn.Module

        """
        super().__init__()
        self.backbone_feature_size = encoder_feature_size
        self.disable_fp16_on_heads = disable_fp16_on_heads
        self.fp16_scope = (
            functools.partial(torch.cuda.amp.autocast, enabled=(not disable_fp16_on_heads))
            if disable_fp16_on_heads
            else layers.dummy_context
        )

        # The base network and the head network used for the self-supervised objective
        # TODO(xsuaucuadros): Bake feature size in the actual resnet model
        hack_feature_size = encoder_feature_size != 2048
        if hack_feature_size and not isinstance(encoder, Cifar10_ResNet):
            encoder = nn.Sequential(
                *encoder[:-1],  # assumes no final BN
                layers.BatchNormND(encoder_feature_size, dim=2),
                layers.Squeezer(),
                nn.Linear(encoder_feature_size, encoder_feature_size),
                nn.ReLU(),
                layers.BatchNormND(encoder_feature_size, dim=1),
            )
        self.base_network = encoder

        # The NCE head.
        self.head = nn.Sequential(
            nn.Linear(encoder_feature_size, head_latent_size),
            layers.BatchNormND(head_latent_size, dim=1),
            nn.ReLU(),
            nn.Linear(head_latent_size, head_latent_size),
            layers.BatchNormND(head_latent_size, dim=1),
            nn.ReLU(),
            nn.Linear(head_latent_size, head_output_size),
            layers.BatchNormND(head_output_size, dim=1),
        )

        # Register all the task head networks.
        self.heads = nn.ModuleDict(heads)
        if hack_feature_size:
            num_classes = len(self.heads["linear_classifier"].bias)
            for k, m in self.heads.items():
                self.heads[k] = nn.Linear(encoder_feature_size, num_classes)

    def to_jit_script(self, model_file: str):
        """Convert a model to jit script and save it."""
        # Convert SyncBN --> BN2d and BN1d
        base_network = layers.BatchNormND.revert_sync_batchnorm(self.base_network)

        # JIT the network and save it
        network = torch.jit.script(base_network)
        torch.jit.save(network, model_file)

    def forward_features(
        self, images: t.Union[torch.Tensor, t.Tuple[torch.Tensor]]
    ) -> torch.Tensor:
        """Extract features from a single of multi-tuple of inputs."""

        # Standard feature extraction pipeline.
        return self.base_network(images).view(-1, self.backbone_feature_size)

    def _post_process_representation(
        self,
        representation1: torch.Tensor,
        representation2: torch.Tensor,
        **extra_results_to_return,
    ) -> t.Dict[str, torch.Tensor]:
        """Process representation through nce projection net and head nets."""
        with self.fp16_scope():
            # convert reprs to fp32 if we are disabling fp16 on heads.
            representation1 = (
                representation1.float() if self.disable_fp16_on_heads else representation1
            )
            representation2 = (
                representation2.float() if self.disable_fp16_on_heads else representation2
            )

            # infer NCE head network
            logits_for_nce1 = self.head(representation1)
            logits_for_nce2 = self.head(representation2)

            # Stop-gradients to the classifier to not learn a trivially better model.
            repr_to_classifier = representation1.detach().clone()  # always detach!
            head_output_dict = {
                head_name: head(repr_to_classifier) for head_name, head in self.heads.items()
            }

        return {
            "nce_logits1": logits_for_nce1,
            "nce_logits2": logits_for_nce2,
            "representations1": repr_to_classifier,
            "representations2": representation2.detach().clone(),
            **head_output_dict,
            **extra_results_to_return,
        }

    def forward(
        self,
        inputs: t.Dict[str, t.Union[torch.Tensor, t.Tuple[torch.Tensor]]],
        **unused_kwargs,
    ) -> t.Dict:
        """Returns the NCE logits and the head predictions."""
        augmentation1, augmentation2 = layers.multicrop_to_split_augmentations(
            inputs, self.training
        )
        if augmentation2 is None:  # at test we only have one aug
            augmentation2 = augmentation1

        representation1 = self.forward_features(augmentation1)
        representation2 = self.forward_features(augmentation2)
        return self._post_process_representation(representation1, representation2)

    def minibatch_metrics(self, **unused_kwargs) -> t.Dict[str, t.Any]:
        """Metrics that are presented at the minibatch level."""
        metrics = utils.recurse_for_attr(self.base_network, "minibatch_metrics")
        result = {}
        for m in metrics:
            result.update({f"{k}_scalar": v for k, v in m.items()})
        return result

    def epoch_metrics(self, **unused_kwargs) -> t.Dict[str, torch.Tensor]:
        """Metrics that are presented at the epoch level."""
        return dict()

    def post_backward_callable(self):
        """Method apply after gradient step."""
        pass
