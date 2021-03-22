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

        # variables
        self.samples = sample_size
        self.std_level = std_level

        self.device = device

    def forward(self, x, target_cls=None, sec_ord=False):
        if target_cls == None:
            target_cls = self.net(x).max(1)[1].item()

        self._reset(x, sec_ord)

        batch_x = self._generate_noised_x(x)
        accus = self._predict(batch_x)

        sal = grad(accus[:, target_cls].mean(), x, create_graph=sec_ord)[0]
        accu = self._predict(x)
        return sal, accu

    def _reset(self, x, sec_ord):
        self.x_std = (x.max() - x.min()) * self.std_level
        if not sec_ord:
            x.requires_grad_(True)

    def _predict(self, x):
        return F.softmax(self.net(x), 1)

    def _generate_noised_x(self, x):
        noise = torch.empty(
            self.samples, self.x_shape[1], self.x_shape[2], self.x_shape[3]).normal_(0, self.x_std.item())
        return x + noise.to(self.device)
