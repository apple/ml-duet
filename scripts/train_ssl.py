#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
import os
import pathlib
import pprint
import typing as t
from time import time

try:
    import wandb

    has_wandb_module = True
except ModuleNotFoundError:
    has_wandb_module = False

import yaml
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim import AdamW

from duet.augmentations import CODE_AUG_MAP, identity_tensor_augmentations

from duet.models.encoders import cifar10_resnet
from duet.models.duet import DUET
from duet.models.essl import ESSL

from duet.objectives.duet import nt_xent_duet
from duet.objectives.essl import essl_loss
from duet.objectives.classification import softmax_cross_entropy
from duet.objectives.common import LOSS_TO_OPTIMIZE
from duet.objectives.schedulers import CosineAnnealingLRWarmup

from parser import train_ssl_parser

# todo: remove ssl hack
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def _load_wandb_credentials() -> bool:
    wandb_file = pathlib.Path(".wandb.yaml")
    try:
        with wandb_file.open("r") as fp:
            wandb_data = yaml.load(fp, Loader=yaml.FullLoader)
        os.environ["WANDB_BASE_URL"] = wandb_data["WANDB_BASE_URL"]
        os.environ["WANDB_API_KEY"] = wandb_data["WANDB_API_KEY"]
        # os.environ["WANDB_MODE"] = "offline"
        return True
    except:
        print(f"Could not configure WandB from data in {wandb_file}.")
        return False


def setup_wandb(args: argparse.Namespace) -> bool:
    """
    Starts wandb if `args.wandb_project` is set.

    :param args: The user defined arguments.
    :return: True if wandb was properly initialized.
    """
    if args.wandb_project is not None:
        assert has_wandb_module, "To use wandb, please `pip install wandb`."
        assert _load_wandb_credentials()
        try:
            wandb.init(project=args.wandb_project, config=vars(args))
            return True
        except Exception as exc:
            print(exc)
            return False


def build_dataset(
    args: argparse.Namespace,
    shuffle_test: bool = False,
    center_crop: bool = False,
) -> t.Tuple[DataLoader, DataLoader, int]:
    """
    Builds the train and test datasets (loaders).

    Note that we pass `pre_tx` as transforms, which basically resize and convert to Tensor.
    The remaining transformations will be applied inside each model's `forward()`
    and are user defined in `args.tx`.

    :param args: The user defined arguments.
    :param shuffle_test: If True, test dataset is als shuffled
    :param center_crop: If True, images are cropped omiting a boundary.
                        Usually boosts test performance, but removes information.

    :return: A tuple (train_loader, test_loader, number of channels)
    """
    pre_tx = identity_tensor_augmentations(
        image_size_override=args.image_size, center_crop=center_crop
    )
    if args.dataset == "mnist":
        dataset_train = torchvision.datasets.MNIST(
            root=args.dataset_root,
            train=True,
            download=True,
            transform=pre_tx["train_transform"],
        )
        dataset_test = torchvision.datasets.MNIST(
            root=args.dataset_root,
            train=False,
            download=True,
            transform=pre_tx["test_transform"],
        )
        channels = 1
    elif args.dataset == "cifar10":
        dataset_train = torchvision.datasets.CIFAR10(
            root=args.dataset_root,
            train=True,
            download=True,
            transform=pre_tx["train_transform"],
        )
        dataset_test = torchvision.datasets.CIFAR10(
            root=args.dataset_root,
            train=False,
            download=True,
            transform=pre_tx["test_transform"],
        )
        channels = 3
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported.")

    train_loader = DataLoader(
        dataset=dataset_train, shuffle=True, batch_size=args.batch_size, pin_memory=True
    )
    test_loader = DataLoader(
        dataset=dataset_test, shuffle=shuffle_test, batch_size=args.batch_size, pin_memory=True
    )
    return train_loader, test_loader, channels


def loss_function(
    args, preds: t.Dict[str, torch.Tensor], training: bool
) -> t.Dict[str, torch.Tensor]:
    """
    Computes the losses required to train each model.

    :param args: The user defined arguments.
    :param preds: Dict produced by each model's `forward()`, contains various model predictions of interest.
    :param training: Are we in training mode (True) or test (False).

    :return: A dict with various losses. The main loss has key `LOSS_TO_OPTIMIZE`.
    """
    if args.model == "duet":
        loss = nt_xent_duet(
            embedding1=preds["nce_logits1"],
            embedding2=preds["nce_logits2"],
            joint_logits1=preds["joint_logits1"],
            joint_logits2=preds["joint_logits2"],
            aug1_params=preds["aug1_params"],
            aug2_params=preds["aug2_params"],
            lambd=args.duet_lambda,
            capsule_dim=args.duet_bins,
            sigma=args.duet_sigma,
            groups=[
                args.equi_tx,
            ],
            target=args.duet_target,
            training=training,
        )
    elif args.model == "simclr":
        loss = nt_xent_duet(
            embedding1=preds["nce_logits1"],
            embedding2=preds["nce_logits2"],
            joint_logits1=preds["joint_logits1"],
            joint_logits2=preds["joint_logits2"],
            aug1_params=preds["aug1_params"],
            aug2_params=preds["aug2_params"],
            lambd=0,
            capsule_dim=0,
            sigma=args.duet_sigma,
            groups=[],
            target=args.duet_target,
            training=training,
        )
    elif args.model == "essl":
        loss = essl_loss(
            embedding1=preds["nce_logits1"],
            embedding2=preds["nce_logits2"],
            essl_logits=preds["essl_logits"],
            essl_labels=preds["essl_labels"],
            lambd=args.essl_lambda,
            groups=[
                args.equi_tx,
            ],
        )
    else:
        raise NotImplementedError(f"Loss for model {args.model} not implemented.")
    return loss


def train_epoch(
    args: argparse.Namespace,
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> pd.DataFrame:
    """
    Update model over one epoch and compute train loss and extra data on the train dataset.

    :return: A pd.DataFrame with the collected data for all steps in the epoch (all batches).
    """
    epoch_results = []
    model.train()
    for step, (imgs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs = imgs.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        inputs = {"unaugmented": imgs}
        preds = model(inputs)
        ssl_loss = loss_function(args=args, preds=preds, training=True)
        tracking_head_loss = softmax_cross_entropy(
            logits=preds["linear_classifier"],
            labels=labels,
            batch_size=args.batch_size,
        )
        loss = ssl_loss[LOSS_TO_OPTIMIZE] + tracking_head_loss[LOSS_TO_OPTIMIZE]
        loss.backward()
        optimizer.step()
        epoch_results.append(
            {
                "total_loss": loss.item(),
                **{"model_" + k: v.item() for k, v in ssl_loss.items()},
                **{"track_head_" + k: v.item() for k, v in tracking_head_loss.items()},
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if args.debug_steps is not None and step == args.debug_steps:
            break

    return pd.DataFrame(data=epoch_results)


def test(
    args: argparse.Namespace,
    model: torch.nn.Module,
    test_loader: DataLoader,
) -> pd.DataFrame:
    """
    Compute test loss and extra data on the test dataset.

    :return: A pd.DataFrame with the collected data for all steps in the epoch (all batches).
    """
    epoch_results = []
    model.eval()
    with torch.no_grad():
        for step, (imgs, labels) in enumerate(test_loader):
            imgs = imgs.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            inputs = {"unaugmented": imgs}
            preds = model(inputs)
            ssl_loss = loss_function(args=args, preds=preds, training=False)
            tracking_head_loss = softmax_cross_entropy(
                logits=preds["linear_classifier"],
                labels=labels,
                batch_size=args.batch_size,
            )
            loss = ssl_loss[LOSS_TO_OPTIMIZE] + tracking_head_loss[LOSS_TO_OPTIMIZE]
            epoch_results.append(
                {
                    "total_loss": loss.item(),
                    **{"model_" + k: v.item() for k, v in ssl_loss.items()},
                    **{"track_head_" + k: v.item() for k, v in tracking_head_loss.items()},
                }
            )
            if args.debug_steps is not None and step == args.debug_steps:
                break

    return pd.DataFrame(data=epoch_results)


def build_model(args: argparse.Namespace, channels: int) -> torch.nn.Module:
    """
    Build the desired model among "duet", "simclr" and "essl"; as specified in args.model.

    :param args: The passed arguments.

    :return: A nn.Module.
    """
    encoder = getattr(cifar10_resnet, args.encoder)(n_input_channels=channels)
    heads = {"linear_classifier": torch.nn.Linear(512, args.num_classes)}
    if args.model == "duet":
        model = DUET(
            encoder_feature_size=512,
            head_output_size=128,
            head_latent_size=2048,
            image_size_override=args.image_size,
            encoder=encoder,
            crop_scale=(0.2, 1.0),
            heads=heads,
            augs=args.tx,
            groups=[
                args.equi_tx,
            ],
            capsule_dim=args.duet_bins,
        )
    elif args.model == "simclr":
        model = DUET(
            encoder_feature_size=512,
            head_output_size=128,
            head_latent_size=2048,
            image_size_override=args.image_size,
            encoder=encoder,
            crop_scale=(0.2, 1.0),
            heads=heads,
            augs=args.tx,
            groups=[],
            capsule_dim=0,
        )
    elif args.model == "essl":
        model = ESSL(
            encoder_feature_size=512,
            head_output_size=128,
            head_latent_size=2048,
            image_size_override=args.image_size,
            encoder=encoder,
            crop_scale=(0.2, 1.0),
            heads=heads,
            augs=args.tx,
            groups=[
                args.equi_tx,
            ],
            capsule_dim=args.duet_bins,
            lambd=args.essl_lambda,
        )
    else:
        raise NotImplementedError(f"Model {args.model} not implemented.")
    return model


def _df_to_string(df: pd.Series, training: bool) -> str:
    cyan = "\033[0;36m"
    green = "\033[0;32m"
    color = cyan if training else green
    nocolor = "\033[0m"
    texts = [f"{c}: {df[c]:0.3f}" for c in df.keys()]
    return color + ", ".join(texts) + nocolor


def log_epoch_mean_data(
    df: pd.DataFrame, epoch: int, tic: float, training: bool, has_wandb: bool
) -> None:
    """
    Logs average data in a df containing all steps data (eg. for a given epoch)

    :param df: The DataFrame with the per-step data.
    :param epoch: Current epoch.
    :param tic: Time since epoch started.
    :param training: Whether we are logging training or test data.
    :param has_wandb: If wandb is ready, pass True.
    """
    df_mean = df.mean()
    print(
        f"[{'Train' if training else 'Test '} epoch {epoch} in {time() - tic:0.2f}s]",
        _df_to_string(df_mean, training=training),
    )
    if has_wandb:
        to_log = {f"{'train' if training else 'test'}_" + c: df_mean[c] for c in df_mean.keys()}
        to_log["epoch"] = epoch
        wandb.log(to_log, step=epoch)


def save_model(
    args: argparse.Namespace, model: torch.nn.Module, epoch: t.Union[str, int], prefix: str
) -> None:
    if not args.save_dir:
        return
    else:
        args.save_dir.mkdir(exist_ok=True, parents=True)
    tx_str = args.equi_tx
    save_name = f"{prefix}_{args.dataset}_{args.model}_{tx_str}"
    if args.model == "duet":
        save_name += f"_eq-{args.equi_tx}_target-{args.duet_target}"
    if args.model == "essl":
        save_name += f"_eq-{args.equi_tx}"
    save_name += f"_{epoch:04d}.th" if isinstance(epoch, int) else f"_{epoch}.th"
    torch.save(model, args.save_dir / save_name)
    print(f"Model saved as {args.save_dir / save_name}")


if __name__ == "__main__":
    args = train_ssl_parser()

    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb if requested
    has_wandb = setup_wandb(args)

    if args.model == "essl":
        # ESSL requires tx and equi_tx to be disjoint!
        # fmt: off
        new_tx = set(args.tx) - set([args.equi_tx,])  # noqa
        # fmt: on
        if len(new_tx) != len(args.tx):
            print(f"[ESSL] Transformation {args.equi_tx} removed from tx, only using {new_tx}")
            args.tx = list(new_tx)

    # Choose default DUET sigma according to the target
    setattr(args, "duet_sigma", 0.2 if args.duet_target == "ga" else 3.0)
    pprint.pprint(vars(args), indent=4)

    train_loader, test_loader, channels = build_dataset(args, center_crop=args.center_crop)
    model = build_model(args, channels=channels).to(args.device)

    optimizer = AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0001)
    scheduler = CosineAnnealingLRWarmup(optimizer, T_max=args.epochs, T_warmup=args.warmup_epochs)

    print("Training starts!")
    for epoch in range(args.epochs):
        tic = time()
        df_epoch_train = train_epoch(
            args=args,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
        )
        log_epoch_mean_data(
            df=df_epoch_train, epoch=epoch, tic=tic, training=True, has_wandb=has_wandb
        )

        if epoch % args.test_every == 0 or epoch == args.epochs - 1:
            tic = time()
            df_epoch_test = test(
                args=args,
                model=model,
                test_loader=test_loader,
            )
            log_epoch_mean_data(
                df=df_epoch_test, epoch=epoch, tic=tic, training=False, has_wandb=has_wandb
            )
            save_model(args=args, model=model, epoch=epoch, prefix="encoder")

        scheduler.step()

    save_model(args=args, model=model, epoch="final", prefix="encoder")
