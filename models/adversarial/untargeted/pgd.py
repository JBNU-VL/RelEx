import torch
from torch import nn
import numpy as np
from cleverhans.future.torch.attacks import projected_gradient_descent


class ProjectedGradientDescent:
    def __init__(self, net, eps=0.07, a=0.01, K=40, norm=np.inf,
                 max_min_bounds=None):
        self.net = net

        # variables
        self.eps = eps
        self.a = a
        self.K = K
        self.norm = norm
        self.max_min_bounds = max_min_bounds

    def __call__(self, x, eps=None):
        if eps is None:
            eps = self.eps
        return pgd_fn(self.net, x, eps, self.a, self.K, self.norm,
                      self.max_min_bounds[0], self.max_min_bounds[1])


def pgd_fn(net, x, eps=0.07, a=0.01, K=40, norm=np.inf, max_bound=None,
           min_bound=None):
    from cleverhans.future.torch.attacks import projected_gradient_descent
    adv_x = projected_gradient_descent(
        model_fn=net,
        x=x,
        eps=eps,
        eps_iter=a,
        nb_iter=K,
        norm=norm,
        clip_max=max_bound,
        clip_min=min_bound,
        sanity_checks=False
    )
    return adv_x
