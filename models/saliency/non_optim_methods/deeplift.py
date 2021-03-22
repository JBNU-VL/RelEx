import torch
from torch import nn
from torch.nn import functional as F
from captum.attr import DeepLift


class DeepLIFT(nn.Module):
    def __init__(self, net):
        super().__init__()

        self.net = net
        self.captum_dl = DeepLift(nn.Sequential(net, nn.Softmax(1)))

    def forward(self, x, target_cls=None):
        if target_cls == None:
            target_cls = self.net(x).max(1)[1].item()

        sal = self.captum_dl.attribute(x, target=target_cls)
        accu = self._predict(x)
        return sal, accu

    def _predict(self, x):
        return F.softmax(self.net(x), 1)
