import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad

from utils import clamp


class ManipulationMethod(nn.Module):
    def __init__(self, net, opts):
        super.__init__()
        self.net = net

        # hyper parameters
        self.lr = opts.lr
        self.max_iters = opts.structured.max_iters

        self.criterion = Loss(opts.structured.factors)

    def forward(self, x, target_x, sal_method):
        adv_x = self._reset(x)

        target_sal = sal_method(target_x, act_fn='softplus', beta_val=1000)
        orig_acc = self._predict(x)[0]

        current_beta_val = self.start_beta_val
        for i in range(self.max_iters):
            adv_sal, adv_acc = sal_method(adv_x, sec_ord=True, act_fn='softplus',
                                          beta_val=current_beta_val)
            loss = self.criterion(adv_sal, target_sal, adv_acc, orig_acc)
            loss.backward()
            self._step()
            adv_x = clamp(adv_x.data, self.mean, self.std)

        return adv_x

    def _reset(self, x):
        adv_x = x.detach().clone().requires_grad_(True)
        self.optimizer = torch.optim.Adam([adv_x], lr=self.lr)
        return adv_x

    def _predict(self, x):
        return self.net(x).max(dim=-1)

    def _step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


class Loss(nn.Module):
    def __init__(self, factors):
        self.factors = factors

    def forward(self, adv_sal, target_sal, adv_acc, orig_acc):
        h_loss_term = self.factors[0] * F.mse_loss(adv_sal, target_sal)
        g_loss_term = self.factors[1] * F.mse_loss(adv_acc, orig_acc)
        return h_loss_term + g_loss_term
