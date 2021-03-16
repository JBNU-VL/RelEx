import torch
from torch import nn
import numpy as np

from .util import replace_activation2softplus, replace_activation2relu, image_clamp


class IterativeAttack(nn.Module):
    def __init__(self, method='topk', eps=1. / 255 / 0.225, k=1000,
                 num_iters=100, alpha=1, measurement='intersection',
                 beta_growth=False, x_bounds=None, beta_range=None):
        super().__init__()
        if method not in ('mass_center', 'topk', 'random', 'target'):
            raise ValueError(f'method `{method}` not supported.')
        if measurement not in ('intersection', 'correlation', 'mass_center'):
            raise ValueError(f'measurement `{measurement}` not supported.')

        self.method = method  # 'mass_center' or 'topk' or 'random' or 'target'
        self.num_iters = num_iters  # 100
        self.k = k  # 1000

        self.alpha = alpha  # 1.
        self.eps = eps
        self.measurement = measurement  # 'intersection' or 'correlation' or 'mass_center'

        self.beta_growth = beta_growth  # True or False

        self.x_bounds = x_bounds
        self.beta_range = beta_range

    def forward(self, x, sal_method, eps=None, target_map=None):
        self.sal_method = sal_method
        sal, orig_accu = self.sal_method(x)
        sal = self._normalize(sal).detach()
        self.orig_sal = sal = sal.flatten()
        target_cls = orig_accu.max(1)[1].detach()

        if self.beta_growth:
            replace_activation2softplus(self.sal_method, beta=10.)
        else:
            replace_activation2softplus(self.sal_method, beta=30.)

        self.mass_center_orig = self._mass_center(x, target_cls)
        self.topk_mask = torch.zeros_like(sal).flatten()
        _, topk_indices = torch.topk(sal, self.k)
        self.topk_mask[topk_indices] = 1.

        min_criterion = 1.
        if eps is not None:
            self.eps = eps
        _adv_x = self._reset(x.detach())
        adv_x = None

        for i in range(self.num_iters):
            if self.beta_growth:
                current_beta = self._calc_beta(i)
            else:
                current_beta = 30.

            perturbation = self.perturb(
                _adv_x, target_cls, target_map, beta=current_beta)
            _adv_x.data = self._step(_adv_x.data, perturbation)
            criterion = self.check_measure(_adv_x.detach(), target_cls)

            if criterion < min_criterion:
                print(
                    f'iter: {i}, criterion: {criterion}, current_beta: {current_beta}')
                min_criterion = criterion
                adv_x = _adv_x.clone().detach()

                if min_criterion == 0:
                    break

        if min_criterion == 1.:
            adv_x = _adv_x.detach().clone()

        adv_sal, adv_accu = self.sal_method(adv_x, target_cls)
        return adv_x, adv_sal, adv_accu

    def _reset(self, x):
        self.inf_norm_bounds = (x + self.eps, x - self.eps)
        adv_x = x.detach().clone().requires_grad_(True)
        return adv_x

    def _step(self, adv_x, perturbation):
        adv_x = adv_x + self.alpha * perturbation
        adv_x = image_clamp(adv_x, self.inf_norm_bounds)
        adv_x = image_clamp(adv_x, self.x_bounds)
        return adv_x

    def _calc_beta(self, i):
        return self.beta_range[0] * (self.beta_range[1] / self.beta_range[0]) ** (i / self.num_iters)

    def _normalize(self, sal):
        sal_norm = torch.abs(sal).sum(dim=1, keepdim=True)
        sal_norm = sal_norm / sal_norm.sum()
        return sal_norm

    def topk_direction(self, adv_x, target_cls):
        adv_sal = self.sal_method(adv_x, target_cls, sec_ord=True)[0]
        adv_sal = self._normalize(adv_sal)

        adv_sal_flatten = torch.flatten(adv_sal)
        loss = (adv_sal_flatten * self.topk_mask.clone().detach()).sum()
        return -torch.autograd.grad(loss, adv_x)[0]

    def _mass_center(self, x, target_cls):
        _, _, w, h = x.shape
        sal = self.sal_method(x, target_cls, sec_ord=True)[0]
        sal = self._normalize(sal)

        x_mesh, y_mesh = torch.meshgrid(torch.arange(w), torch.arange(h))
        x_mesh = x_mesh.type(x.type())
        y_mesh = y_mesh.type(x.type())
        mass_center = torch.stack([
            (sal * x_mesh).sum() / (w * h),
            (sal * y_mesh).sum() / (w * h)
        ])
        return mass_center

    def mass_center_direction(self, adv_x, target_cls):
        mass_center_original = self.mass_center_original
        mass_center_perturbed = self._mass_center(adv_x, target_cls)
        loss = -((mass_center_perturbed - mass_center_original)**2).sum()
        return -torch.autograd.grad(loss, adv_x)[0]

    def target_direction(self, adv_x, target_cls, target_map):
        target_map = target_map.clone().detach()
        adv_sal = self.sal_method(adv_x, target_cls, sec_ord=True)[0]
        adv_sal = -self._normalize(adv_sal)
        loss = -(adv_sal * target_map).sum()
        return torch.autograd.grad(loss, adv_x)[0]

    def perturb(self, adv_x, target_cls, target_map=None, beta=30.):
        replace_activation2softplus(self.sal_method, beta=beta)

        if self.method == 'random':
            perturbation = torch.normal(0, 1, adv_x.shape)
            perturbation = perturbation.type(adv_x.type())
        elif self.method == 'topk':
            perturbation = self.topk_direction(adv_x, target_cls)
        elif self.method == 'mass_center':
            perturbation = self.mass_center_direction(
                adv_x, target_cls)
        elif self.method == 'target':
            perturbation = self.target_direction(
                adv_x, target_cls, target_map)
        return torch.sign(perturbation)

    def check_measure(self, adv_x, target_cls):
        replace_activation2relu(self.sal_method)
        adv_sal, adv_accu = self.sal_method(adv_x, target_cls)
        adv_sal = self._normalize(adv_sal)
        adv_cls = adv_accu.max(1)[1]

        if adv_cls == target_cls:
            if self.measurement == 'intersection':
                _, top_indices = torch.topk(adv_sal.flatten(), self.k)

                _, orig_top_indices = torch.topk(
                    self.orig_sal, self.k
                )
                criterion = float(len(np.intersect1d(
                    top_indices.detach().cpu().numpy(),
                    orig_top_indices.detach().cpu().numpy()
                ))) / self.k
            elif self.measurement == 'correlation':
                flatten = adv_sal.flatten()
                criterion = scipy.stats.spearmanr(
                    self.sal1_flatten.clone().detach().cpu().numpy(),
                    flatten.cpu().numpy())
            elif self.measurement == 'mass_center':
                center = self._mass_center(adv_x, target_cls)
                criterion = -np.linalg.norm(
                    self.mass_center_original.clone().detach().cpu().numpy(),
                    center.cpu().numpy()
                )
            return criterion
        return 1.
