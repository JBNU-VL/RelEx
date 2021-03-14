import torch
from torch import nn
import torch.nn.functional as F
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

    def forward(self, x, target_cls=None, sec_ord=False):
        self._reset(x)
        x.requires_grad_(True)
        batch_x = self._generate_noised_x(x)
        outputs = self.net(batch_x)

        if target_cls == None:
            with torch.no_grad():
                target_cls = self.net(x).max(1)[1].item()

        sal = grad(outputs[:, target_cls].mean(), x, create_graph=sec_ord)[0]
        accu = F.softmax(outputs, 1)
        return sal, accu

    def _reset(self, x):
        self.x_std = (x.max() - x.min()) * self.std_level

    def _generate_noised_x(self, x):
        noise = torch.empty(
            self.samples, self.x_shape[1], self.x_shape[2], self.x_shape[3]).normal_(0, self.x_std)
        return x + noise.to(self.device)
