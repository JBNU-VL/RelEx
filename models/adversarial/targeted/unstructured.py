import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad


class IterativeAttack(nn.Module):
    def __init__(self, net, opts):
        self.net = net
        self.softplus_net =

        # 'random' or 'topk' or 'mass_center' or 'target'
        if opts.unstructured.method not in ('mass_center', 'topk', 'random', 'target'):
            raise ValueError(
                f'method `{opts.unstructured.method}` is not supported.')
        self.method = opts.unstructured.method

        self.max_iters = opts.unstructured.max_iters
        pass

    def forward(self, x, sal_method):
        for i in range(self.max_iters):
            pass
        pass

    def
