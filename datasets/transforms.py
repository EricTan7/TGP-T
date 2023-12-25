from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop, ColorJitter,
    RandomApply, GaussianBlur, RandomGrayscale, RandomResizedCrop,
    RandomHorizontalFlip, RandomErasing
)
from torchvision.transforms.functional import InterpolationMode

AVAI_CHOICES = [
    "random_flip",
    "random_resized_crop",
    "normalize",
    "random_crop",
    "center_crop",  # This has become a default operation during testing
    "colorjitter",
    "randomgrayscale",
    "gaussian_blur",
    "randomerasing",
    "randaug"
]

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


def build_transform(cfg, is_train=True, choices=None):
    """Build transformation function.

    Args:
        cfg (CfgNode): config.
        is_train (bool, optional): for training (True) or test (False).
            Default is True.
        choices (list, optional): list of strings which will overwrite
            cfg.INPUT.TRANSFORMS if given. Default is None.
    """
    if cfg.INPUT.NO_TRANSFORM:
        print("Note: no transform is applied!")
        return None

    if choices is None:
        choices = cfg.INPUT.TRANSFORMS

    for choice in choices:
        assert choice in AVAI_CHOICES

    target_size = f"{cfg.INPUT.SIZE[0]}x{cfg.INPUT.SIZE[1]}"

    normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    if is_train:
        return _build_transform_train(cfg, choices, target_size, normalize)
    else:
        return _build_transform_test(cfg, choices, target_size, normalize)


def _build_transform_train(cfg, choices, target_size, normalize):
    print("Building transform_train")
    tfm_train = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
    input_size = cfg.INPUT.SIZE

    # Make sure the image size matches the target size
    conditions = []
    conditions += ["random_crop" not in choices]
    conditions += ["random_resized_crop" not in choices]
    if all(conditions):
        print(f"+ resize to {target_size}")
        tfm_train += [Resize(input_size, interpolation=interp_mode)]

    if "random_crop" in choices:
        crop_padding = cfg.INPUT.CROP_PADDING
        print(f"+ random crop (padding = {crop_padding})")
        tfm_train += [RandomCrop(input_size, padding=crop_padding)]

    if "random_resized_crop" in choices:
        s_ = cfg.INPUT.RRCROP_SCALE
        print(f"+ random resized crop (size={input_size}, scale={s_})")
        tfm_train += [
            RandomResizedCrop(input_size, scale=s_, interpolation=interp_mode)
        ]

    if "random_flip" in choices:
        print("+ random flip")
        tfm_train += [RandomHorizontalFlip()]

    if "colorjitter" in choices:
        b_ = cfg.INPUT.COLORJITTER_B
        c_ = cfg.INPUT.COLORJITTER_C
        s_ = cfg.INPUT.COLORJITTER_S
        h_ = cfg.INPUT.COLORJITTER_H
        print(
            f"+ color jitter (brightness={b_}, "
            f"contrast={c_}, saturation={s_}, hue={h_})"
        )
        tfm_train += [
            ColorJitter(
                brightness=b_,
                contrast=c_,
                saturation=s_,
                hue=h_,
            )
        ]

    if "randomgrayscale" in choices:
        print("+ random gray scale")
        tfm_train += [RandomGrayscale(p=cfg.INPUT.RGS_P)]

    if "gaussian_blur" in choices:
        print(f"+ gaussian blur (kernel={cfg.INPUT.GB_K})")
        gb_k, gb_p = cfg.INPUT.GB_K, cfg.INPUT.GB_P
        tfm_train += [RandomApply([GaussianBlur(gb_k)], p=gb_p)]

    print("+ to torch tensor of range [0, 1]")
    tfm_train += [ToTensor()]

    if "normalize" in choices:
        print(
            f"+ normalization (mean={cfg.INPUT.PIXEL_MEAN}, std={cfg.INPUT.PIXEL_STD})"
        )
        tfm_train += [normalize]

    if "randomerasing" in choices:
        print("+ random erasing")
        tfm_train += [RandomErasing()]

    tfm_train = Compose(tfm_train)

    return tfm_train


def _build_transform_test(cfg, choices, target_size, normalize):
    print("Building transform_test")
    tfm_test = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
    input_size = cfg.INPUT.SIZE

    print(f"+ resize the smaller edge to {max(input_size)}")
    tfm_test += [Resize(max(input_size), interpolation=interp_mode)]

    print(f"+ {target_size} center crop")
    tfm_test += [CenterCrop(input_size)]

    print("+ to torch tensor of range [0, 1]")
    tfm_test += [ToTensor()]

    if "normalize" in choices:
        print(
            f"+ normalization (mean={cfg.INPUT.PIXEL_MEAN}, std={cfg.INPUT.PIXEL_STD})"
        )
        tfm_test += [normalize]

    tfm_test = Compose(tfm_test)

    return tfm_test