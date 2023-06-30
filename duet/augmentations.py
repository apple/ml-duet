#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

"""
SSL augmentation stacks with gettable params.
"""

import typing as t
import math
from PIL import Image
from torchvision import transforms
import kornia.augmentation as K
from kornia.geometry import get_rotation_matrix2d
import torch
from torch import nn
from torch.nn import functional as F
import tree
from copy import deepcopy


CODE_AUG_MAP = {
    "sca": "RandomResizedCrop",
    "rot": "RandomRotation",
    "ron": "RandomRotation90",
    "fli": "RandomHorizontalFlip",
    "flv": "RandomVerticalFlip",
    "blu": "RandomGaussianBlur",
    "gra": "RandomGrayscale",
    "jit": "ColorJitter",
    "res": "Resize",
    "ccr": "CenterCrop",
}

DEFAULT_CROP_SCALE = (0.2, 1.0)
DEFAULT_CROP_PERCENTAGE = 0.875


def identity_tensor_augmentations(
    image_size_override: t.Optional[int] = 32,
    center_crop: bool = True,
) -> t.Dict[str, t.Callable]:
    """Identity augmentation that only converts to tensor.

    :returns: train_transforms, test_transforms
    :rtype: list, list

    """
    if center_crop:
        expand = 256 / 224
        first_test_resize = int(math.floor(image_size_override * expand))
        train_transform = transforms.Compose(
            [
                transforms.Resize(first_test_resize, interpolation=Image.BILINEAR),
                transforms.CenterCrop(first_test_resize),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(first_test_resize, interpolation=Image.BILINEAR),
                transforms.CenterCrop(first_test_resize),
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(image_size_override, interpolation=Image.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(image_size_override, interpolation=Image.BILINEAR),
                transforms.ToTensor(),
            ]
        )
    return {"train_transform": train_transform, "test_transform": test_transform}


def aug_stack_from_codes(
    codes: t.List[str],
) -> t.Tuple[t.List[str], t.List[str], t.Dict[str, float]]:
    """
    Builds an augmentation stack (just aug names) from a list of codes.
    Color transforms will be collapsed into ColorJitter,
    placed where the first color transform is observed in codes.

    :param codes: The augmentation codes.
    :return: Train augs, test augs, color jitter params.
    """

    color_jitter_params = {
        "brightness": 0.8 if "bri" in codes else 0.0,
        "contrast": 0.8 if "con" in codes else 0.0,
        "saturation": 0.8 if "sat" in codes else 0.0,
        "hue": 0.2 if "hue" in codes else 0.0,
    }
    color_codes = ["bri", "con", "sat", "hue"]
    codes_modif = deepcopy(codes)
    for color_tx in color_codes:
        if color_tx in codes_modif:
            if "jit" not in codes_modif:
                # Swap first occuring color tx by jitter.
                idx = codes_modif.index(color_tx)
                codes_modif.remove(color_tx)
                codes_modif.insert(idx, "jit")
            else:
                # Remove all others.
                codes_modif.remove(color_tx)

    train_stack_names = [CODE_AUG_MAP[code] for code in codes_modif]
    train_stack_names.append("Normalize")
    test_stack_names = [
        # "Resize",
        "CenterCrop",
        "Normalize",
    ]
    return train_stack_names, test_stack_names, color_jitter_params


class RandomRotation90(K.AugmentationBase2D):
    def __init__(self) -> None:
        super(RandomRotation90, self).__init__(p=1.0)

    def generate_parameters(self, input_shape: torch.Size):
        # generate the random parameters for your use case.
        batch_shape = int(input_shape[0])
        angles_deg = torch.randint(0, 4, size=(batch_shape,)) * 90
        return dict(degrees=angles_deg.to(torch.float32))

    def compute_transformation(
        self,
        input: torch.Tensor,
        params: t.Dict[str, torch.Tensor],
        **unused_kwargs,
    ) -> torch.Tensor:
        bs, _, h, w = input.shape
        center = torch.zeros((bs, 2))
        center[:, 0] = w / 2
        center[:, 1] = h / 2
        angle = params["degrees"].to(input.dtype)
        scale = torch.ones_like(center).to(input.dtype)
        rot_mat: torch.tensor = get_rotation_matrix2d(center, angle, scale)  # 1x2x3
        return rot_mat

    def apply_transform(
        self,
        input: torch.Tensor,
        params: t.Dict[str, torch.Tensor],
        transform: t.Optional[torch.Tensor] = None,
        **unused_kwargs,
    ):
        # apply transformation and return
        _, _, h, w = input.shape
        angles = params["degrees"]
        input[angles == 90] = input[angles == 90].flip(3).transpose(2, 3)
        input[angles == 180] = input[angles == 180].flip(3).flip(2)
        input[angles == 270] = input[angles == 270].transpose(2, 3).flip(3)
        return input


class ParamAugStack(nn.Module):
    """Augmentations with gettable transforms params."""

    def __init__(
        self,
        image_size_override: int,
        crop_scale: t.Tuple[float, float] = DEFAULT_CROP_SCALE,
        color_jitter_strength: float = 0.5,
        random_resize_crop_embed: str = "rescale",
        normalize_color_augs: bool = True,
        rotation_degrees: int = 180,
        normalize_rotation: bool = True,
        aug_codes: t.List[str] = ["sca"],
    ):
        super().__init__()
        if crop_scale is None:
            crop_scale = DEFAULT_CROP_SCALE

        (
            self.train_stack_names,
            self.test_stack_names,
            color_jitter_params,
        ) = aug_stack_from_codes(aug_codes)

        self.color_jitter_params = {
            "brightness": 0.8,
            "contrast": 0.8,
            "saturation": 0.8,
            "hue": 0.2,
        }
        if color_jitter_params is not None:
            self.color_jitter_params.update(color_jitter_params)

        self.color_jitter_params = tree.map_structure(
            lambda x: color_jitter_strength * x, self.color_jitter_params
        )

        self.image_size_override = image_size_override
        self.normalize_color_augs = normalize_color_augs
        self.normalize_rotation = normalize_rotation
        self.rotation_degrees = rotation_degrees
        self.crop_scale = crop_scale

        assert random_resize_crop_embed in ("raw", "rescale")
        self.random_resize_crop_embed = random_resize_crop_embed

        self.train_augs = nn.ModuleDict(
            {
                "Resize": K.Resize(
                    size=(image_size_override, image_size_override),
                    resample="bilinear",
                ),
                "RandomResizedCrop": K.RandomResizedCrop(
                    size=(image_size_override, image_size_override),
                    scale=crop_scale,
                    resample="bilinear",
                ),
                "RandomRotation": K.RandomRotation(degrees=self.rotation_degrees, p=1.0),
                "RandomRotation90": RandomRotation90(),
                "RandomHorizontalFlip": K.RandomHorizontalFlip(p=0.5),
                "RandomVerticalFlip": K.RandomVerticalFlip(p=0.5),
                "ColorJitter": K.ColorJitter(
                    p=0.8,
                    **self.color_jitter_params,
                ),
                "RandomGaussianBlur": K.RandomGaussianBlur(
                    kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5
                ),
                "RandomSolarize": K.RandomSolarize(p=0.2, thresholds=0.1, additions=0.1),
                "RandomGrayscale": K.RandomGrayscale(p=0.2),
                "Normalize": K.Normalize(
                    mean=0.5,
                    std=1.0,
                ),
                "CenterCrop": K.CenterCrop(size=image_size_override),
            }
        )
        self.test_augs = nn.ModuleDict(
            {
                "Resize": K.Resize(
                    size=(image_size_override, image_size_override),
                    resample="bilinear",
                ),
                "CenterCrop": K.CenterCrop(size=image_size_override),
                "Normalize": K.Normalize(
                    mean=0.5,
                    std=1.0,
                ),
            }
        )

    def make_default_dict(self) -> t.Dict[str, torch.Tensor]:
        defaults_non_normalized = {
            "RandomResizedCrop": [
                0,
                0,
                self.image_size_override,
                self.image_size_override,  # x, y, h, w
            ],
            "CenterCrop": [
                0,
                0,
                self.image_size_override,
                self.image_size_override,  # x, y, h, w
            ],
            "ColorJitter": [
                1,
                1,
                1,
                0,  # b, c, s, h
                1,
                0,
                0,
                0,  # order param 0 in dims 4,5,6,7
                0,
                1,
                0,
                0,  # order param 1 in dims 8,9,10,11
                0,
                0,
                1,
                0,  # order param 2 in dims 12,13,14,15
                0,
                0,
                0,
                1,  # order param 3 in dims 16,17,12,19
            ],
            "RandomHorizontalFlip": [0],
            "RandomVerticalFlip": [0],
            "RandomGaussianBlur": [0],
            "RandomGrayscale": [0],
            "RandomSolarize": [0.5, 0],
            "RandomRotation": [0],
            "RandomRotation90": [0],
        }

        defaults_normalized = {
            "RandomResizedCrop": [
                0,
                0,
                1,
                1,  # x, y, h, w
            ],
            "CenterCrop": [
                0,
                0,
                1,
                1,  # x, y, h, w
            ],
            "ColorJitter": [
                0.5,
                0.5,
                0.5,
                0.5,  # b, c, s, h
                1,
                0,
                0,
                0,  # order param 0 in dims 4,5,6,7
                0,
                1,
                0,
                0,  # order param 1 in dims 8,9,10,11
                0,
                0,
                1,
                0,  # order param 2 in dims 12,13,14,15
                0,
                0,
                0,
                1,  # order param 3 in dims 16,17,12,19
            ],
            "RandomHorizontalFlip": [0.25],
            "RandomVerticalFlip": [0.25],
            "RandomGaussianBlur": [0],
            "RandomGrayscale": [0],
            "RandomSolarize": [0.5, 0.5],
            "RandomRotation": [0.5],
            "RandomRotation90": [0.125],
        }

        out_dict = defaults_normalized if self.normalize_color_augs else defaults_non_normalized
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in out_dict.items()}

    def _color_jitter_params(self, xform: t.Callable) -> torch.Tensor:
        """
        Extract the parameters used for Kornia ColorJitter.

        The parameters are uniformly sampled in the ranges:
        * brightness (1-b, 1+b)
        * contrast (1-c, 1+c)
        * saturation (1-s, 1+s)
        * hue (-h, h)

        So the default value (no aug applied) should be all 1, but 0 for hue.

        The default order in Kornia is 0, 1, 2, 3.

        :returns A tensor (B, Params) where those rows where the xform was not applied are set to 0.
        """
        params = xform._params
        assert params is not None, f"Need to apply the {xform._get_name()} xform first."
        applied_mask = params["batch_prob"]  # (B, )
        batch_size = len(applied_mask)
        applied_params = torch.stack(
            [
                params["brightness_factor"],
                params["contrast_factor"],
                params["saturation_factor"],
                params["hue_factor"],
                # TODO(jramapuram): do we need params["order"]?
            ],
        ).T  # (B_applied, P)

        def normalize_params(p: torch.Tensor, jitter: t.Dict[str, float]) -> torch.Tensor:
            """Normalize parameters according to sampling ranges"""
            p[:, 0] = (p[:, 0] - (1 - jitter["brightness"])) / (2 * jitter["brightness"])
            p[:, 1] = (p[:, 1] - (1 - jitter["contrast"])) / (2 * jitter["contrast"])
            p[:, 2] = (p[:, 2] - (1 - jitter["saturation"])) / (2 * jitter["saturation"])
            p[:, 3] = (p[:, 3] - (0 - jitter["hue"])) / (2 * jitter["hue"])
            return p

        if self.normalize_color_augs:
            applied_params = normalize_params(applied_params, self.color_jitter_params)

        defaults = self.make_default_dict()["ColorJitter"].to(applied_params.dtype)
        out_tensor = defaults.repeat(batch_size, 1)

        if len(applied_params) > 0:
            # Get also order, which is the same for all images.
            order = params["order"].repeat(len(applied_params), 1)  # (B_applied, P)
            # Create a binary representation of every ordering usable by a NN
            order_one_hot = F.one_hot(order.reshape(-1, 1), num_classes=4).reshape(
                len(applied_params), -1
            )
            applied_params = torch.cat([applied_params, order_one_hot], 1)  # (B_applied, P + order)

            applied_mask = applied_mask.to(torch.bool)
            out_tensor[applied_mask] = applied_params  # (B, P + order)

        return out_tensor

    def _random_resize_crop_params(self, xform: t.Callable) -> torch.Tensor:
        """
        Extract the parameters used for Kornia RandomResizedCrop.

        NOTE: Assumes cropping is applied with p=1.

        :returns A tensor (B, Params) where those rows where the xform was not applied are set to 0.
        """
        params = xform._params
        assert params is not None, f"Need to apply the {xform._get_name()} xform first."
        applied_mask = params["batch_prob"]  # (B, )
        batch_size = len(applied_mask)
        # Just make it float for now
        src = params["src"].float()

        if self.random_resize_crop_embed in ("raw", "rescale"):
            # Points are sorted (top_left, bottom_left, bottom_right, top_right)
            origin = src[:, 0]
            height = src[:, 1, 0] - src[:, 0, 0]
            width = src[:, 2, 1] - src[:, 1, 1]

            applied_params = torch.cat(
                [origin, height[:, None], width[:, None]], 1
            )  # (B_applied, 4)

            defaults = self.make_default_dict()["RandomResizedCrop"].to(applied_params.dtype)
            out_tensor = defaults.repeat(batch_size, 1)

            if self.random_resize_crop_embed == "rescale":
                applied_params /= self.image_size_override
            applied_mask = applied_mask.to(torch.bool)
            out_tensor[applied_mask] = applied_params  # (B, 4)
        elif self.random_resize_crop_embed == "sincos":
            raise NotImplementedError("sincos not implemented properly in this branch.")
        else:
            raise ValueError(f"Unknown embedding method {self.random_resize_crop_embed}.")

        return out_tensor

    @staticmethod
    def _random_horizontal_flip_params(xform: t.Callable) -> torch.Tensor:
        """
        Extract the parameters used for Kornia RandomHorizontalFlip.

        :returns A tensor (B, Params) where those rows where the xform was not applied are set to 0.
        """
        params = xform._params
        assert params is not None, f"Need to apply the {xform._get_name()} xform first."
        out_tensor = params["batch_prob"].view(-1, 1).to(torch.float)  # (B, 1)
        # Normalize symmetrically around 0.5
        return out_tensor / 2.0 + 0.25

    @staticmethod
    def _random_gaussian_blur_params(xform: t.Callable) -> torch.Tensor:
        """
        Extract the parameters used for Kornia RandomGaussianBlur.

        :returns A tensor (B, Params) where those rows where the xform was not applied are set to 0.
        """
        params = xform._params
        assert params is not None, f"Need to apply the {xform._get_name()} xform first."
        return params["batch_prob"].view(-1, 1).to(torch.float)  # (B, 1)

    @staticmethod
    def _random_grayscale_params(xform: t.Callable) -> torch.Tensor:
        """
        Extract the parameters used for Kornia RandomGrayscale.

        :returns A tensor (B, Params) where those rows where the xform was not applied are set to 0.
        """
        params = xform._params
        assert params is not None, f"Need to apply the {xform._get_name()} xform first."
        return params["batch_prob"].view(-1, 1).to(torch.float)  # (B, 1)

    def _random_solarize_params(self, xform: t.Callable) -> torch.Tensor:
        """
        Extract the parameters used for Kornia RandomSolarize.

        The parameters are uniformly sampled in the ranges:
        * thresholds (0.5-t, 0.5+t)
        * additions (-a, a)

        Therefore, the default values are 0.5 and 0 respectively.

        :returns A tensor (B, Params) where those rows where the xform was not applied are set to 0.
        """
        params = xform._params
        assert params is not None, f"Need to apply the {xform._get_name()} xform first."
        applied_mask = params["batch_prob"]  # (B, )
        batch_size = len(applied_mask)
        applied_params = torch.stack(
            [
                params["thresholds"],
                params["additions"],
            ],
        ).T  # (B_applied, P)

        default = self.make_default_dict()["RandomSolarize"].to(applied_params.dtype)
        out_tensor = default.repeat(batch_size, 1)

        if self.normalize_color_augs:
            # TODO(xsuaucuadros): Assumes thresholds and additions = 0.1
            applied_params[:, 0] = (applied_params[:, 0] - 0.5 + 0.1) / 0.2
            applied_params[:, 1] = (applied_params[:, 1] + 0.1) / 0.2

        applied_mask = applied_mask.to(torch.bool)
        out_tensor[applied_mask] = applied_params  # (B, P)
        return out_tensor

    def _random_rotation_params(self, xform: t.Callable) -> torch.Tensor:
        """
        Extract the parameters used for Kornia RandomRotation.

        The parameters are uniformly sampled in the ranges:
        * degrees (-self.rotation_degrees, self.rotation_degrees)

        Therefore, the default values are 0.5 (normalized) and 0 respectively.

        :returns A tensor (B, Params) where those rows where the xform was not applied are set to 0.
        """
        params = xform._params
        assert params is not None, f"Need to apply the {xform._get_name()} xform first."
        applied_mask = params["batch_prob"]  # (B, )
        batch_size = len(applied_mask)
        applied_params = params["degrees"].view(-1, 1)  # (B_applied, P=1)

        default = self.make_default_dict()["RandomRotation"].to(applied_params.dtype)
        out_tensor = default.repeat(batch_size, 1)

        if self.normalize_rotation:
            # Angle is rotated between -a, +a
            applied_params = (applied_params + self.rotation_degrees) / (2 * self.rotation_degrees)

        applied_mask = applied_mask.to(torch.bool)
        out_tensor[applied_mask] = applied_params  # (B, P)

        return out_tensor

    def _random_rotation90_params(self, xform: t.Callable) -> torch.Tensor:
        """
        Extract the parameters used for custom RandomRotation90.

        The parameters angles either 0,90, 180, 270

        Therefore, the default values are 0 (normalized) and 0 respectively.

        :returns A tensor (B, Params) where those rows where the xform was not applied are set to 0.
        """
        params = xform._params
        assert params is not None, f"Need to apply the {xform._get_name()} xform first."
        applied_mask = params["batch_prob"]  # (B, )
        batch_size = len(applied_mask)
        applied_params = params["degrees"].view(-1, 1)  # (B_applied, P=1)

        default = self.make_default_dict()["RandomRotation90"].to(applied_params.dtype)
        out_tensor = default.repeat(batch_size, 1)

        if self.normalize_rotation:
            # Angle is rotated between 0, 90, 180, 270 degrees.
            # Normalize so it's symmetric around 0.5.
            applied_params = applied_params / 360.0 + 0.125

        applied_mask = applied_mask.to(torch.bool)
        out_tensor[applied_mask] = applied_params  # (B, P)

        return out_tensor

    def forward(
        self, images: torch.Tensor, force_eval: bool = False
    ) -> t.Tuple[torch.Tensor, t.Optional[torch.Tensor]]:
        """
        Performs data augmentation according to the stack selected.

        :param images: Input images (B, C, H, W)
        :param force_eval: forces evaluation aug stack.
        :return: A tuple (augmented_images, augmentation_params). The augmentation_params are (B, P),
        with all params of all augmentations concatenated in each row.
        """
        device = images.device
        aug_callers = {
            "ColorJitter": self._color_jitter_params,
            "RandomResizedCrop": self._random_resize_crop_params,
            "CenterCrop": self._random_resize_crop_params,
            "RandomHorizontalFlip": self._random_horizontal_flip_params,
            "RandomVerticalFlip": self._random_horizontal_flip_params,
            "RandomGaussianBlur": self._random_gaussian_blur_params,
            "RandomGrayscale": self._random_grayscale_params,
            "RandomSolarize": self._random_solarize_params,
            "RandomRotation": self._random_rotation_params,
            "RandomRotation90": self._random_rotation90_params,
        }

        def _call_stack(aug_stack_names, aug_stack):
            """Execute a stack and return."""
            out = images
            aug_params = {}
            for aug_name in aug_stack_names:
                augmentation = aug_stack[aug_name]
                # Apply augmentation to data
                out = augmentation(out)
                # Get augmentation params
                if aug_name in aug_callers:
                    aug_i = aug_callers[aug_name](augmentation)
                    aug_params[aug_name] = aug_i

            aug_params = tree.map_structure(lambda x: x.cuda() if images.is_cuda else x, aug_params)
            return out, aug_params

        if self.training and not force_eval:
            train_variates, train_aug_params = _call_stack(self.train_stack_names, self.train_augs)
            train_aug_params = tree.map_structure(lambda x: x.to(device), train_aug_params)
            if "CenterCrop" in train_aug_params:
                train_aug_params["RandomResizedCrop"] = train_aug_params.pop("CenterCrop")
            return train_variates, train_aug_params

        # Test case
        test_variates, test_aug_params = _call_stack(self.test_stack_names, self.test_augs)
        if "CenterCrop" in test_aug_params:
            test_aug_params["RandomResizedCrop"] = test_aug_params.pop("CenterCrop")
        test_batch_size = len(test_variates)
        default_dict = self.make_default_dict()
        for train_aug_name in self.train_stack_names:
            if train_aug_name not in test_aug_params and train_aug_name in aug_callers:
                aug_default = default_dict[train_aug_name].repeat(test_batch_size, 1)
                test_aug_params[train_aug_name] = aug_default

        test_aug_params = tree.map_structure(lambda x: x.to(device), test_aug_params)

        return test_variates, test_aug_params
