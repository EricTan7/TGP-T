from datasets import DataManager
from processor import do_train
from tools.utils import set_random_seed
from tools.logger import setup_logger
from tools.train_utils import *

import argparse
import torch
import torch.distributed as dist
import os
import wandb
import warnings
warnings.filterwarnings("ignore")


def main(args):
    cfg = setup_cfg(args)
    set_random_seed(cfg.SEED)

    if cfg.WANDB:
        run = wandb.init(project=args.wandb_proj, config=cfg, tags=["TGP-T"])
        run.name = f'{cfg.DATASET.NAME}-{cfg.DATASET.NUM_SHOTS}s'

    output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME, f"{cfg.DATASET.NUM_SHOTS}shots")
    logger = setup_logger(cfg.TRAINER.NAME, output_dir, if_train=True)

    if cfg.TRAIN.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    # Dataset
    data = DataManager(cfg)

    # Model
    model = MODELS[cfg.TRAINER.NAME](cfg, data.dataset.classnames)

    # Train
    do_train(cfg, model, data, output_dir, args.local_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="./runs", help="output directory")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument(
        "--dist-train", type=bool, default=False, help="path to config file"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="configs/configs/imagenet.yaml", help="path to config file"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=16,
        help="number of shots",
    )
    parser.add_argument("--trainer", type=str, default="TGPT", help="name of trainer")  # CoOp
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="use wandb to log runs"
    )
    parser.add_argument(
        "--wandb-proj", type=str, default="TGPT", help="project name of wandb"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
