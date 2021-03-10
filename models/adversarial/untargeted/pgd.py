import torch
from torch import nn
import numpy as np

from cleverhans.future.torch.attacks import projected_gradient_descent, spsa

r'''
PGD class module
'''


class ProjectedGradientDescent(nn.Module):
    def __init__(self, net, opts):
        super.__init__()
        pass

    def forward(self, x):
        pass


def pgd(x, net, opts):
    x_adv = projected_gradient_descent(model_fn=net, x=x, eps=opts.pgd.eps,
                                       eps_iter=opts.pgd.a, nb_iter=opts.pgd.K,
                                       norm=np.inf, clip_max=opts.img_max_val,
                                       clip_min=opts.img_min_val,
                                       sanity_checks=False)
    return x_adv
