import torch
from torch import nn
import torch.nn.functional as F
from .util import replace_activation2softplus, replace_activation2relu, image_clamp

r'''
'''


class ManipulationMethod(nn.Module):
    def __init__(self, lr=0.0002, num_iters=1500, factors=(1e11, 1e6),
                 beta_range=(10., 100.), x_max_min_bounds=None, device=None):
        super().__init__()
        # hyper parameters
        self.lr = lr
        self.num_iters = num_iters
        self.beta_range = beta_range

        self.criterion = Loss(factors)

        self.x_max_min_bounds = (
            x_max_min_bounds[0].to(device), x_max_min_bounds[1].to(device)
        )

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

        replace_activation2relu(sal_method)
        adv_sal = sal_method(adv_x.detach())[0]
        return adv_x, adv_sal, orig_sal, target_sal

    def _reset(self, x):
        adv_x = x.detach().clone().requires_grad_(True)
        self.optimizer = torch.optim.Adam([adv_x], lr=self.lr)
        return adv_x

    def _step(self, adv_x):
        self.optimizer.step()
        self.optimizer.zero_grad()
        image_clamp(adv_x, self.x_max_min_bounds)

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
