#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

"""
Code from: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

Refer to `ACKNOWLEDGEMENTS.MD` for extra details on code reproducibility and license.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from duet.layers import Identity, View

from torch.autograd import Variable

__all__ = [
    "Cifar10_ResNet",
    "cifar10_resnet20",
    "cifar10_resnet32",
    "cifar10_resnet44",
    "cifar10_resnet56",
    "cifar10_resnet110",
    "cifar10_resnet1202",
    "lifted_cifar10_resnet20",
    "lifted_cifar10_resnet32",
    "lifted_cifar10_resnet44",
    "lifted_cifar10_resnet56",
    "lifted_cifar10_resnet110",
    "lifted_cifar10_resnet1202",
]


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A", normalize_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = normalize_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = normalize_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    normalize_layer(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Cifar10_ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        n_input_channels=3,
        disable_backbone_final_bn=False,
        normalize_layer=nn.BatchNorm2d,
        **unused_kwargs,
    ):
        super(Cifar10_ResNet, self).__init__()

        # Small size as in original repo (for supervised learning), only get ~70% SimCLR Downstream Test Acc.
        # self.in_planes = 16
        # self.conv1 = nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = normalize_layer(16)
        # self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # Big size, (made by me) reaches ~90% downstream test acc.
        self.in_planes = 64
        self.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = normalize_layer(64)
        self.layer1 = self._make_layer(
            block, 128, num_blocks[0], stride=1, normalize_layer=normalize_layer
        )
        self.layer2 = self._make_layer(
            block, 256, num_blocks[1], stride=2, normalize_layer=normalize_layer
        )
        self.layer3 = self._make_layer(
            block, 512, num_blocks[2], stride=2, normalize_layer=normalize_layer
        )

        # Remove final Linear Layer, we use the backbone for feature extraction only.
        # self.linear = nn.Linear(64, num_classes)

        self.final_bn = Identity()
        if not disable_backbone_final_bn:
            self.final_bn = nn.Sequential(
                # Small size (has 64 output dim, expansion=1)
                # View([-1, 64 * block.expansion, 1, 1]),
                # normalize_layer(64 * block.expansion),
                # Big size (has output size 512, expansion=1)
                View([-1, 512 * block.expansion, 1, 1]),
                normalize_layer(512 * block.expansion),
            )

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, normalize_layer):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # For small size, use option 'A'
            # layers.append(block(self.in_planes, planes, stride))

            # For big size use option 'B' for shortcut layers instead of 'A' default
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    option="B",
                    normalize_layer=normalize_layer,
                )
            )
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = torch.flatten(out, 1)
        out = self.final_bn(out)
        # out = self.linear(out)

        return out


### Original RGB Models
def cifar10_resnet20(**kwargs):
    return Cifar10_ResNet(BasicBlock, [3, 3, 3], **kwargs)


def cifar10_resnet32(**kwargs):
    return Cifar10_ResNet(BasicBlock, [5, 5, 5], **kwargs)


def cifar10_resnet44(**kwargs):
    return Cifar10_ResNet(BasicBlock, [7, 7, 7], **kwargs)


def cifar10_resnet56(**kwargs):
    return Cifar10_ResNet(BasicBlock, [9, 9, 9], **kwargs)


def cifar10_resnet110(**kwargs):
    return Cifar10_ResNet(BasicBlock, [18, 18, 18], **kwargs)


def cifar10_resnet1202(**kwargs):
    return Cifar10_ResNet(BasicBlock, [200, 200, 200], **kwargs)


### Lifted Models (currently hardcoded to 54 channels -- 4 scale, 4 orientation + lowpass + highpass)
def lifted_cifar10_resnet20(**kwargs):
    return Cifar10_ResNet(BasicBlock, [3, 3, 3], n_input_channels=54, **kwargs)


def lifted_cifar10_resnet32(**kwargs):
    return Cifar10_ResNet(BasicBlock, [5, 5, 5], n_input_channels=54, **kwargs)


def lifted_cifar10_resnet44(**kwargs):
    return Cifar10_ResNet(BasicBlock, [7, 7, 7], n_input_channels=54, **kwargs)


def lifted_cifar10_resnet56(**kwargs):
    return Cifar10_ResNet(BasicBlock, [9, 9, 9], n_input_channels=54, **kwargs)


def lifted_cifar10_resnet110(**kwargs):
    return Cifar10_ResNet(BasicBlock, [18, 18, 18], n_input_channels=54, **kwargs)


def lifted_cifar10_resnet1202(**kwargs):
    return Cifar10_ResNet(BasicBlock, [200, 200, 200], n_input_channels=54, **kwargs)


def test(net):
    import numpy as np

    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print(
        "Total layers",
        len(
            list(
                filter(
                    lambda p: p.requires_grad and len(p.data.size()) > 1,
                    net.parameters(),
                )
            )
        ),
    )


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith("cifar10_resnet") or net_name.startswith("lifted_cifar10_resnet"):
            print(net_name)
            test(globals()[net_name]())
            print()
