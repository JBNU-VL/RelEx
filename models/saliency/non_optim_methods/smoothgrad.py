import torch
from torch import nn
from torch.autograd import grad


class SmoothGrad(nn.Module):
    def __init__(self, net, shape=(1, 3, 224, 224), sample_size=50,
                 std_level=0.1, device=None):
        super().__init__()

        self.net = net

        self.x_shape = shape
        self.samples = sample_size

        self.std_level = std_level

        self.device = device

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

        return x + noise.to(self.device)
