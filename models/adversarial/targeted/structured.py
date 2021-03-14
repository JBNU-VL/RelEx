import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad

r'''
'''


class ManipulationMethod(nn.Module):
    def __init__(self, lr=0.0002, num_iters=1500, factors=(1e11, 1e6),
                 beta_range=(10., 100.), bounds=None):
        super().__init__()
        # hyper parameters
        self.lr = lr
        self.num_iters = num_iters
        self.beta_range = beta_range

        self.criterion = Loss(factors)

        self.bounds = bounds

    def forward(self, orig_x, target_x, sal_method):
        replace_activation2softplus(sal_method, beta=1000.)
        target_sal, target_accu = sal_method(target_x)
        target_sal_norm = self._normalize(target_sal)
        orig_sal, orig_accu = sal_method(orig_x)

        adv_x = self._reset(orig_x)

        for i in range(self.num_iters):
            # apply to replace relu with softplus.
            # change bata valu in softplus.
            current_beta = self._calc_beta(i)
            replace_activation2softplus(sal_method, beta=current_beta)

            adv_sal, adv_accu = sal_method(adv_x, sec_ord=True)
            adv_sal_norm = self._normalize(adv_sal)

            loss = self.criterion(
                adv_sal_norm, target_sal_norm, adv_accu, orig_accu)
            loss.backward(retain_graph=True)
            self._step(adv_x.data)

        adv_sal = sal_method(adv_x.detach())[0]
        return adv_x, adv_sal, orig_sal, target_sal

    def _reset(self, x):
        adv_x = x.detach().clone().requires_grad_(True)
        self.optimizer = torch.optim.Adam([adv_x], lr=self.lr)
        return adv_x

    def _step(self, adv_x):
        self.optimizer.step()
        self.optimizer.zero_grad()
        image_clamp(adv_x, self.bounds)

    def _calc_beta(self, i):
        return self.beta_range[0] * (self.beta_range[1] / self.beta_range[0]) ** (i / self.num_iters)

    def _normalize(self, sal):
        sal_norm = torch.abs(sal).sum(dim=1)
        sal_norm = sal_norm / sal_norm.sum()
        return sal_norm


class Loss(nn.Module):
    def __init__(self, factors):
        super().__init__()
        self.factors = factors

    def forward(self, adv_sal, target_sal, adv_acc, orig_acc):
        h_loss_term = self.factors[0] * F.mse_loss(adv_sal, target_sal)
        g_loss_term = self.factors[1] * F.mse_loss(adv_acc, orig_acc)
        return h_loss_term + g_loss_term


def replace_activation2softplus(net, beta=10.):
    if net.__class__.__name__ != 'RealTimeSaliency':
        for child_name, child in net.named_children():
            if isinstance(child, nn.ReLU) or isinstance(child, nn.Softplus):
                setattr(net, child_name, nn.Softplus(beta=beta))
            else:
                replace_activation2softplus(child, beta)


def image_clamp(adv_x, bounds):
    assert bounds[0].size() == (
        1, 3, 224, 224), "max bound shape is not correct."
    assert bounds[1].size() == (
        1, 3, 224, 224), "min bound shape is not correct."
    max_bound, min_bound = bounds
    adv_x = torch.where(adv_x > max_bound, adv_x, max_bound)
    adv_x = torch.where(adv_x < min_bound, adv_x, min_bound)
