import sys
sys.path.insert(0, '.')

from models import TGPT_Model, TGPT_lora_Model
from configs import get_cfg_default
import logging


MODELS = {
    'TGPT': TGPT_Model,
    'TGPT_lora': TGPT_lora_Model
}


def print_args(args, cfg):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    logger.info("***************")
    logger.info("** Arguments **")
    logger.info("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        logger.info("{}: {}".format(key, args.__dict__[key]))
    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.dist_train:
        cfg.TRAIN.DIST_TRAIN = args.dist_train

    if args.shots:
        cfg.DATASET.NUM_SHOTS = args.shots
    
    if args.use_wandb:
        cfg.WANDB = args.use_wandb


def setup_cfg(args):
    cfg = get_cfg_default()

    # 1. From the config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 2. From input arguments
    reset_cfg(cfg, args)

    # 3. From optional input arguments
    cfg.merge_from_list(args.opts)

    return cfg