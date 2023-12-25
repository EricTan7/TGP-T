import logging
import os
import sys
import os.path as osp
import time


def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    # 将日志输出到控制台的handler：ch
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)  # 高于debug级别的日志都要输出
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 将日志输出到文件的handler：fh
    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        if if_train:
            fpath = osp.join(save_dir, "train_log.txt")
            if osp.exists(fpath):
                # make sure the existing log file is not over-written
                file = 'train_log_' + time.strftime("%Y-%m-%d-%H-%M-%S") + '.txt'
                fpath = osp.join(save_dir, file)
            fh = logging.FileHandler(fpath, mode='a')
        else:
            fpath = osp.join(save_dir, "test_log.txt")
            if osp.exists(fpath):
                file = 'test_log_' + time.strftime("%Y-%m-%d-%H-%M-%S") + '.txt'
                fpath = osp.join(save_dir, file)
            fh = logging.FileHandler(fpath, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
