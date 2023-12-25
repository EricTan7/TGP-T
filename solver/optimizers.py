import warnings
import torch
import torch.nn as nn

AVAI_OPTIMS = ["adam", "amsgrad", "sgd", "rmsprop", "radam", "adamw"]


def build_optimizer(model, optim_cfg, param_groups=None, is_params=False, is_split=False):
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module or list): model.
                can be either:(1) nn.module  (2) list [nn.module, nn.module]
                              (3) list(dict)  [{"params": params, "lr": lr}]
        optim_cfg (CfgNode): optimization config.
        param_groups: If provided, directly optimize param_groups and abandon model
    """
    optim = optim_cfg.NAME
    if is_split:
        lr = optim_cfg.LR_LORA
    else:
        lr = optim_cfg.LR
    weight_decay = optim_cfg.WEIGHT_DECAY
    momentum = optim_cfg.MOMENTUM
    sgd_dampening = optim_cfg.SGD_DAMPNING
    sgd_nesterov = optim_cfg.SGD_NESTEROV
    rmsprop_alpha = optim_cfg.RMSPROP_ALPHA
    adam_beta1 = optim_cfg.ADAM_BETA1
    adam_beta2 = optim_cfg.ADAM_BETA2
    base_lr_mult = optim_cfg.BASE_LR_MULT

    if optim not in AVAI_OPTIMS:
        raise ValueError(
            f"optim must be one of {AVAI_OPTIMS}, but got {optim}"
        )

    params = []

    if is_params:
        param_groups = model
    elif isinstance(model, list):
        # for sub_model in model:
        #     for name, module in sub_model.named_children():
        #         params += [p for p in module.parameters()]
        if isinstance(model[0], dict):  # set lr for different groups
            param_groups = model
        else:
            param_groups = []
            for sub in model:
                # for p in sub.parameters():
                #     if p.requires_grad:
                #         param_groups += p
                param_groups += sub.parameters()
    else:
        # for name, module in model.named_children():
        #     params += [p for p in module.parameters()]
        param_groups = model.parameters()

    # the value in the dict will cover other parameters
    # param_groups = [{"params": params, "lr": lr}]

    if optim == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == "amsgrad":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif optim == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            eps=1e-7
        )
    else:
        raise NotImplementedError(f"Optimizer {optim} not implemented yet!")

    return optimizer