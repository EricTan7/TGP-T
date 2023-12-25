import sys
sys.path.insert(0, '.')
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, classnames, clip_model, logit_scale, bias=False):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)

        self.fc = nn.Linear(vis_dim*2, n_cls, bias=bias)
        self.logit_scale = logit_scale

    def forward(self, x):
        x = F.normalize(x, dim=1)
        x = self.fc(x)
        x = x * self.logit_scale.exp()

        return x