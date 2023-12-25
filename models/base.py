import os.path as osp
from collections import OrderedDict
import torch
import torch.nn as nn

from tools.utils import tolist_if_not
from tools.model import save_checkpoint, load_checkpoint


class BaseModel(nn.Module):
    """
    Basic Model
    Implementation of some common functions
    """
    def __init__(self):
        super().__init__()
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def register_model(self, name="model", model=None, optim=None, sched=None):
        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]

    def get_specific_lr(self, names=None):
        if names is None:
            names = self.get_model_names(names)
            name = names[0]
        else:
            name = names
        return self._optims[name].param_groups[0]["lr"]

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def save_model(self, epoch, directory, is_best=False,
                   val_result=None, model_name=""):
        # save registered_module
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            save_dict = OrderedDict()
            for k, v in self._models[name].named_parameters():
                if v.requires_grad:
                    save_dict[k] = model_dict[k]

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            sdir = osp.join(directory, name)
            save_checkpoint(
                {
                    "state_dict": save_dict,
                    "epoch": epoch,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                sdir,
                is_best=is_best,
                model_name=model_name,
            )
            if not is_best:
                self.logger.info(f"Checkpoint saved to {sdir}")
            else:
                self.logger.info('Best checkpoint saved to "{}"'.format(sdir))

    def load_model(self, directory, epoch=None):
        if not directory:
            self.logger.info("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            self.logger.info("Loading weights to {} " 'from "{}"'.format(name, model_path))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)