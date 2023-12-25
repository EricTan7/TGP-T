import pickle
import shutil
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import logging

from .utils import mkdir_if_missing


def save_checkpoint(
    state,
    save_dir,
    is_best=False,
    remove_module_from_keys=True,
    model_name=""
):
    r"""Save checkpoint.

    Args:
        state (dict): dictionary.
        save_dir (str): directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is True.
        model_name (str, optional): model name to save.
    """
    mkdir_if_missing(save_dir)

    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        state["state_dict"] = new_state_dict

    # save model
    epoch = state["epoch"]
    if not model_name:
        model_name = "model.pth.tar-" + str(epoch)
    fpath = osp.join(save_dir, model_name)

    # save current model name
    # checkpoint_file = osp.join(save_dir, "checkpoint")
    # checkpoint = open(checkpoint_file, "w+")
    # checkpoint.write("{}\n".format(osp.basename(fpath)))
    # checkpoint.close()

    if is_best:
        best_fpath = osp.join(osp.dirname(fpath), "model-best.pth.tar")
        # shutil.copy(fpath, best_fpath)
        torch.save(state, best_fpath)
    else:
        torch.save(state, fpath)


def load_checkpoint(fpath):
    r"""Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        fpath = 'log/my_model/model.pth.tar-10'
        checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint


def load_pretrained_weights(model, weight_path):
    r"""Load pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        # >>> weight_path = 'log/my_model/model-best.pth.tar'
        # >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        print(
            f"Cannot load {weight_path} (check the key names manually)"
        )
    else:
        print(f"Successfully loaded pretrained weights from {weight_path}")
        if len(discarded_layers) > 0:
            print(
                f"Layers discarded due to unmatched keys or size: {discarded_layers}"
            )