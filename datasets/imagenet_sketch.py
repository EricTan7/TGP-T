import os
import pickle
from collections import OrderedDict
from .basic import Benchmark
from tools.utils import listdir_nohidden

from .imagenet import ImageNet


class ImageNetSketch(Benchmark):
    """ImageNet-Sketch.

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-sketch"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data = self.read_data(classnames)

        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = {'impath': impath,
                        'label': int(label),
                        'classname': classname}
                items.append(item)

        return items
