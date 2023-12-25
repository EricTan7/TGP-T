import warnings
import numpy as np
import random
import pickle
import os
import torch
import json
import PIL
from PIL import Image
import errno


def set_random_seed(seed):
    '''Set random seed for reproducibility.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def tolist_if_not(x):
    """Convert to a list."""
    if not isinstance(x, list):
        x = [x]
    return x


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


def check_isfile(fpath):
    """Check if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = os.path.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    return Image.open(path).convert("RGB")


def makedirs(path, verbose=False):
    '''Make directories if not exist.'''
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if verbose:
            print(path + " already exists.")


def save_obj_as_pickle(pickle_location, obj):
    '''Save an object as pickle.'''
    pickle.dump(
        obj, open(pickle_location, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Save object as a pickle at {pickle_location}")


def load_pickle(pickle_location, default_obj=None):
    '''Load a pickle file.'''
    if os.path.exists(pickle_location):
        return pickle.load(open(pickle_location, 'rb'))
    else:
        return default_obj


def save_as_json(obj, fpath):
    '''Save an object as json.'''
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def load_json(json_location, default_obj=None):
    '''Load a json file.'''
    if os.path.exists(json_location):
        try:
            with open(json_location, 'r') as f:
                obj = json.load(f)
            return obj
        except:
            print(f"Error loading {json_location}")
            return default_obj
    else:
        return default_obj


def collect_env_info():
    """Return env info as a string.

    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info

    env_str = get_pretty_env_info()
    env_str += "\n        Pillow ({})".format(PIL.__version__)
    return env_str