from tools.meter import AverageMeter
from tools.dist_utils import reduce_value
from tools.metrics import compute_accuracy
from solver import Classification

import os
import time
import json
import wandb
import logging
import datetime
from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

device = 'cuda'


def do_train(cfg, model, data, output_dir, local_rank):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    optimizer = model.optim
    scheduler = model.sched

    if device:
        model.to(local_rank)
        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    tot_iter = cfg.OPTIM.MAX_ITER

    # 1. loss
    criterion = nn.CrossEntropyLoss()

    # 2. meter
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    category_loss_meter = AverageMeter()
    content_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()

    image_loader_iter = iter(data.train_loader)
    best_val_dict = {
        "iter": 1,
        "val_acc": 0.,
        "model": None
    }

    for iters in range(1, tot_iter+1):
        model.set_model_mode("train")
        start = time.time()
        scheduler.step()
        try:
            image, label, caption = parse_batch_train(next(image_loader_iter))
        except StopIteration:
            image_loader_iter = iter(data.train_loader)
            image, label, caption = parse_batch_train(next(image_loader_iter))

        output, loss_category, loss_content = model(image, label, caption)

        # text supervision loss
        loss_prompts = loss_category + loss_content
        # classification loss
        loss_cls = criterion(output, label)
        # final loss
        loss = loss_cls + loss_prompts

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            loss = reduce_value(loss, average=True)

        # logging
        loss_meter.update(loss.item(), image.shape[0])
        cls_loss_meter.update(loss_cls.item(), image.shape[0])
        category_loss_meter.update(loss_category.item(), image.shape[0])
        content_loss_meter.update(loss_content.item(), image.shape[0])
        acc = compute_accuracy(output, label, topk=(1,))[0].item()
        acc_meter.update(acc, 1)
        batch_time.update(time.time() - start)

        meet_freq = iters % cfg.TRAIN.PRINT_FREQ == 0
        if meet_freq:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                nb_remain = tot_iter - iters
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"Iter [{iters}/{tot_iter}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})"]
                info += [f"cls_loss {cls_loss_meter.val:.3f} ({cls_loss_meter.avg:.3f})"]
                info += [f"category_loss {category_loss_meter.val:.3f} ({category_loss_meter.avg:.3f})"]
                info += [f"content_loss {content_loss_meter.val:.3f} ({content_loss_meter.avg:.3f})"]
                info += [f"acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})"]
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    info += [f"lr {model.module.get_current_lr():.4e}"]
                else:
                    info += [f"lr {model.get_current_lr():.4e}"]
                if cfg.TRAINER.NAME == 'baseline_cattn_vocabloss_shembed_zsinit_lscale_lnable_wiseft_nxcattn':
                    info += [f"lscale {model.model.cls_head.logit_scale:.4e}"]
                info += [f"eta {eta}"]
                logger.info(" ".join(info))

            if cfg.WANDB:
                wandb.log({'train loss': loss_meter.val,
                        'train acc': acc_meter.val,
                        'train cls loss': cls_loss_meter.val,
                        'train category loss': category_loss_meter.val,
                        'train content loss': content_loss_meter.val,
                        'iter': iters
                        })

        # meet iter: save checkpoint
        sdir = output_dir
        if iters % cfg.TRAIN.SAVE_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.save_model(iters, sdir, is_best=False)
                else:
                    model.save_model(iters, sdir, is_best=False)

        # meet iter: do val
        if (iters % cfg.TRAIN.TEST_FREQ == 0):
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.set_model_mode("test")
                    results, val_loss = do_val(cfg, model, data.val_loader)
                    model.module.set_model_mode("train")
                else:
                    model.set_model_mode("test")
                    results, val_loss = do_val(cfg, model, data.val_loader)
                    model.set_model_mode("train")
                
                if results["accuracy"] > best_val_dict["val_acc"]:
                    best_val_dict["iter"] = iters
                    best_val_dict["val_acc"] = results["accuracy"]
                    best_val_dict["model"] = deepcopy(model.state_dict())

                if cfg.WANDB:
                    wandb.log({'val acc': results["accuracy"],
                            'val loss': val_loss,
                            'best val iter': best_val_dict["iter"]
                            })

                    wandb.log({'test acc': 0.,
                            'test acc (wiseft_0.5)': 0.,
                            'test loss': 0.,
                            'test loss (wiseft_0.5)': 0.
                            })

    # do test
    model.load_state_dict(best_val_dict["model"])
    ratio = 0.5
    if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
        model.module.set_model_mode("test")
        results, results_wiseft = do_test(cfg, model, data.test_loader, ratio)
        model.module.set_model_mode("train")
    else:
        model.set_model_mode("test")
        results, results_wiseft = do_test(cfg, model, data.test_loader, ratio)
        model.set_model_mode("train")

    test_results = {
         'test acc': results["accuracy"],
         f'test acc (wiseft_{ratio})': results_wiseft["accuracy"],
    }
    if cfg.WANDB:
        wandb.log(test_results)

    # save the test results
    test_path = os.path.join(output_dir, "test.json")
    with open(test_path, 'w') as f:
        json.dump(test_results, f)

    # save the best model
    if cfg.TRAIN.SAVE_MODEL:
        sdir = output_dir
        model.save_model(0, sdir, is_best=True)

    return test_results


def parse_batch_train(batch):
    input = batch["img"]
    label = batch["label"]
    caption = batch['tokenized_caption']
    input = input.to(device)
    label = label.to(device)
    caption = caption.to(device)
    return input, label, caption


def parse_batch(batch):
    input = batch["img"]
    label = batch["label"]
    input = input.to(device)
    label = label.to(device)
    return input, label


def do_test(cfg, model, test_loader, ratio=0.5):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    evaluator = Classification(cfg, logger)
    evaluator_wiseft = Classification(cfg, logger)

    logger.info(f"Evaluate on the *test* set")
    head = deepcopy(model.model.cls_head.fc)
    zs_weights = deepcopy(model.model.zs_weights)

    wiseft_weights = (1 - ratio) * head.weight.data + ratio * zs_weights
    model.model.wiseft_head.fc.weight.data = wiseft_weights
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            image, label = parse_batch(batch)
            logits, logits_wiseft = model(image)
            evaluator.process(logits, label)
            evaluator_wiseft.process(logits_wiseft, label)

    return evaluator.evaluate(), evaluator_wiseft.evaluate()


def do_val(cfg, model, val_loader):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    evaluator = Classification(cfg, logger)
    evaluator_wiseft = Classification(cfg, logger)
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()

    ratio = 0.5
    logger.info(f"Evaluate on the *val* set")
    head = deepcopy(model.model.cls_head.fc)
    zs_weights = deepcopy(model.model.zs_weights)
    wiseft_weights = (1 - ratio) * head.weight.data + ratio * zs_weights
    model.model.wiseft_head.fc.weight.data = wiseft_weights

    for batch_idx, batch in enumerate(val_loader):
        with torch.no_grad():
            image, label = parse_batch(batch)
            logits, logits_wiseft = model(image)
            loss = criterion(logits, label)
            loss_meter.update(loss.item(), image.shape[0])
            evaluator.process(logits, label)
            evaluator_wiseft.process(logits_wiseft, label)

    return evaluator.evaluate(), loss_meter.avg
