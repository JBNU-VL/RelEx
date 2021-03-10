import torch
from torch import nn
from torch.autograd import grad


class SmoothGrad(nn.Module):
    def __init__(self, net, opts):
        self.net = net

        self.x_shape = opts.x_shape
        self.samples = opts.smoothgrad.samples

        self.std_level = opts.smoothgrad.std_level

    def forward(self, x, target_cls, sec_ord=False):
        self._reset(x)
        x.requires_grad_(True)
        batch_x = self._generate_noised_x(x)
        outputs = self.net(batch_x)[:, target_cls]
        return grad(outputs.mean(), x, create_graph=sec_ord)[0]

    def _reset(self, x):
        self.x_std = (x.max() - x.min()) * self.std_level

    def _generate_noised_x(self, x):
        noise = torch.empty(
            self.samples, self.x_shape[1], self.x_shape[2], self.x_shape[3]).normal_(0, self.x_std)

        return x + noise
