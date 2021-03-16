import torch
from torch import nn

r'''
'''


def replace_activation2softplus(model, beta=10.):
    if model.__class__.__name__ != 'RealTimeSaliency':
        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU) or isinstance(child, nn.Softplus):
                setattr(model, child_name, nn.Softplus(beta=beta))
            else:
                replace_activation2softplus(child, beta)


def replace_activation2relu(model):
    if model.__class__.__name__ != 'RealTimeSaliency':
        for child_name, child in model.named_children():
            if isinstance(child, nn.Softplus):
                setattr(model, child_name, nn.ReLU())
            else:
                replace_activation2relu(child)


def image_clamp(adv_x, bounds):
    assert bounds[0].nelement() == (
        bounds[0] > bounds[1]).sum(), "sequence is wrong."
    assert bounds[0].size() == (
        1, 3, 224, 224), "max bound shape is not correct."
    assert bounds[1].size() == (
        1, 3, 224, 224), "min bound shape is not correct."
    max_bound, min_bound = bounds
    adv_x = torch.where(adv_x > max_bound, max_bound, adv_x)
    adv_x = torch.where(adv_x < min_bound, min_bound, adv_x)
    return adv_x
