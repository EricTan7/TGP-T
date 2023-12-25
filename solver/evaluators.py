import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, logger, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        self.logger = logger
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        info = []
        info += ["=> result"]
        info += [f"* total: {self._total:,}"]
        info += [f"* correct: {self._correct:,}"]
        info += [f"* accuracy: {acc:.1f}%"]
        info += [f"* error: {err:.1f}%"]
        info += [f"* macro_f1: {macro_f1:.1f}%"]
        self.logger.info("\n".join(info))

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            self.logger.info("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                info = []
                info += [f"* class: {label} ({classname})"]
                info += [f"total: {total:,}"]
                info += [f"correct: {correct:,}"]
                info += [f"acc: {acc:.1f}%"]
                self.logger.info("\t".join(info))
            mean_acc = np.mean(accs)
            self.logger.info(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            self.logger.info(f"Confusion matrix is saved to {save_path}")

        return results