import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad


class IntegratedGradient(nn.Module):
    def __init__(self, net, steps=50, device=None):
        super().__init__()

        self.net = net

        # variables
        self.steps = steps if steps != None else 50
        self.pathes = torch.linspace(
            0, 1, self.steps, dtype=torch.float32, device=device)

        self.device = device

    def forward(self, x, target_cls=None, sec_ord=False):
        if target_cls == None:
            target_cls = self.net(x).max(1)[1].item()
        elif isinstance(target_cls, torch.Tensor):
            target_cls = target_cls.item()

        self._reset(x, sec_ord)

        pathes_x = self.pathes.view(self.steps, 1, 1, 1) * x
        accus = self._predict(pathes_x)

        sal = grad(accus[:, target_cls].mean(), x,
                   create_graph=sec_ord)[0] * x.detach()
        accu = self._predict(x)
        return sal, accu

    def _reset(self, x, sec_ord):
        if not sec_ord:
            x.requires_grad_(True)

    def _predict(self, x):
        return F.softmax(self.net(x), 1)
