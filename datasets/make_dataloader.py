import torch
from torch.utils.data import Dataset, DataLoader
from tabulate import tabulate
import logging

from tools.utils import read_image
import torch.distributed as dist

from datasets.transforms import build_transform
from datasets.caltech101 import Caltech101
from datasets.dtd import DescribableTextures
from datasets.eurosat import EuroSAT
from datasets.fgvc_aircraft import FGVCAircraft
from datasets.food101 import Food101
from datasets.imagenet import ImageNet
from datasets.oxford_flowers import OxfordFlowers
from datasets.oxford_pets import OxfordPets
from datasets.stanford_cars import StanfordCars
from datasets.sun397 import SUN397
from datasets.ucf101 import UCF101
from datasets.imagenet_sketch import ImageNetSketch
from datasets.imagenetv2 import ImageNetV2


FACTORY = {
    'Caltech101': Caltech101,
    'DescribableTextures': DescribableTextures,
    'EuroSAT': EuroSAT,
    'FGVCAircraft': FGVCAircraft,
    'Food101': Food101,
    'ImageNet': ImageNet,
    'OxfordFlowers': OxfordFlowers,
    'OxfordPets': OxfordPets,
    'StanfordCars': StanfordCars,
    'SUN397': SUN397,
    'UCF101': UCF101,
    'ImageNetV2': ImageNetV2,
    'ImageNetSketch': ImageNetSketch,
}


def train_collate_fn(batch):
    images, labels = [], []
    for d in batch:
        images.append(d['img'])
        labels.append(torch.tensor(d['label'], dtype=torch.long))

    return torch.stack(images, dim=0), torch.stack(labels, dim=0)


def test_collate_fn(batch):
    images, labels = [], []
    for d in batch:
        images.append(d['img'])
        labels.append(torch.tensor(d['label'], dtype=torch.long))

    return torch.stack(images, dim=0), torch.stack(labels, dim=0)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor

    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index]

    def __len__(self):
        return self.input_tensor.size(0)


class DataManager():
    def __init__(self, cfg, custom_tfm_train=None, custom_tfm_test=None):
        self.logger = logging.getLogger(cfg.TRAINER.NAME)
        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            self.logger.info("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            self.logger.info("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # 1.dataset + transform
        dataset = FACTORY[cfg.DATASET.NAME](cfg)    # dataset.train,  dataset.val,  dataset.test
        train_set = DatasetWrapper(cfg, dataset.train, transform=tfm_train, caption=True)
        val_set = DatasetWrapper(cfg, dataset.val, transform=tfm_test, caption=False)
        test_set = DatasetWrapper(cfg, dataset.test, transform=tfm_test, caption=False)

        # 2.dataloader
        test_batch = cfg.DATALOADER.TEST.BATCH_SIZE
        nw = cfg.DATALOADER.NUM_WORKERS
        if cfg.TRAIN.DIST_TRAIN:
            train_batch = cfg.DATALOADER.TRAIN_X.BATCH_SIZE // dist.get_world_size()
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_loader = DataLoader(train_set,
                                      batch_size=train_batch,
                                      pin_memory=False,
                                      persistent_workers=True,
                                      num_workers=nw,
                                      shuffle=False,
                                      sampler=train_sampler,
                                      drop_last=False)
        else:
            train_batch = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
            train_sampler = torch.utils.data.sampler.RandomSampler(train_set)
            train_loader = DataLoader(train_set,
                                      batch_size=train_batch,
                                      pin_memory=False,
                                      persistent_workers=True,
                                      sampler=train_sampler,
                                      num_workers=nw,
                                      drop_last=False)

        test_loader = DataLoader(test_set,
                                 batch_size=test_batch,
                                 pin_memory=False,
                                 shuffle=False,
                                 num_workers=nw,
                                 drop_last=False)

        val_loader = DataLoader(val_set,
                                batch_size=test_batch,
                                pin_memory=False,
                                persistent_workers=True,
                                shuffle=False,
                                num_workers=nw,
                                drop_last=False)

        # Attributes
        self._num_classes = dataset.num_classes
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME

        table = []
        table.append(["Dataset", dataset_name])

        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train", f"{len(self.dataset.train):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        self.logger.info(tabulate(table))


class DatasetWrapper(Dataset):
    def __init__(self, cfg, data_source, transform=None, caption=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform
        self.use_caption = caption

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        if self.use_caption:
            output = {
                "label": item['label'],
                "impath": item['impath'],
                "index": idx,
                "tokenized_caption": item['tokenized_caption']
            }

        else:
            output = {
                "label": item['label'],
                "impath": item['impath'],
                "index": idx
            }

        img0 = read_image(item['impath'])

        if self.transform is not None:
            output["img"] = self.transform(img0)
        else:
            output["img"] = img0

        return output
