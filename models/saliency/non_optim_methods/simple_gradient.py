import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad


class SimpleGradient(nn.Module):
    def __init__(self, net):
        super().__init__()

        self.net = net

    def forward(self, x, target_cls=None, sec_ord=False):
        if target_cls == None:
            target_cls = self.net(x).max(1)[1].item()
        elif isinstance(target_cls, torch.Tensor):
            target_cls = target_cls.item()

        self._reset(x, sec_ord)
        outputs = self.net(x)

        sal = grad(outputs[:, target_cls].sum(), x,
                   create_graph=sec_ord)[0]
        accu = self._predict(x)
        return sal, accu

    def _reset(self, x, sec_ord):
        if not sec_ord:
            x.requires_grad_(True)

    def _predict(self, x):
        return F.softmax(self.net(x), 1)
