#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
import pathlib
from duet.models.encoders import cifar10_resnet
from duet.augmentations import CODE_AUG_MAP


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="The device, wither cpu or cuda. If None, will use cuda when available.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        metavar="STR",
        choices=["mnist", "cifar10"],
        default="cifar10",
        help="The dataset to use.",
    )
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        metavar="PATH",
        default="/tmp",
        help="Where to save the dataset locally.",
    )
    parser.add_argument(
        "--batch-size", type=int, metavar="INT", default=1024, help="Training batch size."
    )
    parser.add_argument(
        "--image-size",
        type=int,
        metavar="INT",
        default=32,
        help="Input image size. All images will be resized to a square of such size.",
    )
    parser.add_argument(
        "--center-crop",
        action="store_true",
        help="If True, images are cropped at their center, omiting boundaries. Default False.",
    )
    parser.add_argument(
        "--save-dir",
        type=pathlib.Path,
        metavar="PATH",
        default=None,
        help="Where to save the final model.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        metavar="STR",
        default=None,
        help=(
            "Optionally, log to WandB. Requires a file .wandb.yaml with credentials (see"
            " README.md)."
        ),
    )
    return parser


def train_ssl_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        choices=["duet", "simclr", "essl"],
        default="duet",
        help="Model type among duet, simclr or essl.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=cifar10_resnet.__all__,
        default="cifar10_resnet32",
        help="Encoder backbone architecture.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        metavar="INT",
        default=10,
        help="Number of classes in the training set.",
    )
    parser.add_argument("--epochs", type=int, metavar="INT", default=800, help="Training epochs.")
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        metavar="INT",
        default=10,
        help="Learning rate linear warm up in epochs.",
    )
    parser.add_argument(
        "--test-every",
        type=int,
        metavar="INT",
        default=10,
        help="Every how many epochs should the test dataset be evaluated.",
    )
    parser.add_argument(
        "--lr", type=float, metavar="FLOAT", default=0.0004, help="The train learning rate."
    )
    parser.add_argument(
        "--tx",
        type=str,
        nargs="*",
        choices=list(CODE_AUG_MAP.keys()),
        required=True,
        help="Image transformations to be applied, in order of application.",
    )
    parser.add_argument(
        "--equi-tx",
        type=str,
        choices=list(CODE_AUG_MAP.keys()),
        required=True,
        help=(
            "That transformation we want to learn its structure (or we want to become equivariant"
            " to). Must be one of `--tx` for DUET and one not in `--tx` for ESSL."
        ),
    )
    parser.add_argument(
        "--duet-lambda",
        type=float,
        metavar="FLOAT",
        default=10.0,
        help="Lambda parameter in DUET, weighs the structure loss.",
    )
    parser.add_argument(
        "--duet-bins",
        type=int,
        metavar="INT",
        default=8,
        help="Number of bins in the structure axis of DUET, parameter G in the paper.",
    )
    parser.add_argument(
        "--duet-target",
        type=str,
        choices=["ga", "vm"],
        default="ga",
        help=(
            "Target distribution, either `ga` (Gaussian) or `vm` (von-Mises). The parameter \\sigma"
            " in the paper will be automatically chosen accordingly."
        ),
    )
    parser.add_argument(
        "--essl-lambda",
        type=float,
        metavar="FLOAT",
        default=0.4,
        help=(
            "Lambda parameter in ESSL, weighs the equivariance loss. Defaults to 0.4 according to"
            " the ESSL paper."
        ),
    )
    parser.add_argument(
        "--debug-steps",
        type=int,
        metavar="INT",
        default=None,
        help=(
            "Number of steps per epoch, use 2 or 3 to debug and not have to wait for a full epoch."
        ),
    )

    return parser.parse_args()
