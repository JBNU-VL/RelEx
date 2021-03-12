import torch
from torch import nn
import numpy as np

r'''
pgd is referenced by cleverhans.
'''


def pgd_fn(net, x, eps=0.07, a=0.01, K=40, norm=np.inf,
           pre_img_max=(1 - 0.406) / 0.225, pre_img_min=(0 - 0.485) / 0.229):
    from cleverhans.future.torch.attacks import projected_gradient_descent
    x_adv = projected_gradient_descent(model_fn=net,
                                       x=x,
                                       eps=eps,
                                       eps_iter=a,
                                       nb_iter=K,
                                       norm=norm,
                                       clip_max=pre_img_max,
                                       clip_min=pre_img_min,
                                       sanity_checks=False)
    return x_adv
