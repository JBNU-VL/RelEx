import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad


class RelEx(nn.Module):
    def __init__(self, net, shape=(1, 3, 224, 224), batch_size=50, lr=0.1,
                 mtm=0.99, x_std_level=0.1, max_iters=50, lambda1=1e-4,
                 lambda2=1., mode='batch', device=None):
        super().__init__()

        # common variables
        self.x_ch = shape[1]
        self.x_size = shape[-1]

        # hyper-parameters for RelEx
        self.batch_size = batch_size  # 50
        self.lr = lr  # 0.1
        self.mtm = mtm  # 0.99
        self.x_std_level = x_std_level  # 0.1
        self.max_iters = max_iters  # 50 ~ 100
        self.lambda1 = lambda1  # 1e-4
        self.lambda2 = lambda2  # 1.

        # RelEx variables
        self.mode = mode  # 'batch' or 'vanilla'

        self.device = device
        self.jacobian_vector_ones = torch.ones(
            self.batch_size, dtype=torch.float32, device=self.device)
        self.jacobian_vector_weights = self.jacobian_vector_ones / self.batch_size
        # self.jacobian_vector_weights = torch.ones(
        #     self.batch_size, dtype=torch.float32, device=self.device) / self.batch_size

        self.net = net
        self.criterion = Loss(self.lambda1, self.lambda2)

    def forward(self, x, target_cls=None, sec_ord=False):
        m = self._reset(x, target_cls, sec_ord)

        for _ in range(self.max_iters):
            self._generate_grad(x, m)
            self._step(m)

        accu = self._predict(x.detach())
        return m, accu

    def _reset(self, x, target_cls=None, sec_ord=False):
        if target_cls == None:
            target_cls = self.net(x).max(1)[1].item()
        elif isinstance(target_cls, torch.Tensor):
            target_cls = target_cls.item()

        self.target_cls = target_cls

        self.sec_ord = sec_ord  # whether calculate second order derivative

        self.x_std = (x.max() - x.min()) * self.x_std_level  # image std

        # initialize mask
        m = torch.rand(x.size(), device=self.device) / 100
        m.requires_grad_(True)
        self.optimizer = torch.optim.SGD([m], lr=self.lr, momentum=self.mtm)
        return m

    def _generate_grad(self, x, m):
        batch_x = self._generate_noised_x(x)

        foregnd_inputs = batch_x * m
        backgnd_inputs = batch_x * (1 - m)

        foregnd_outputs = self._predict(foregnd_inputs)[:, self.target_cls]
        backgnd_outputs = self._predict(backgnd_inputs)[:, self.target_cls]
        # foregnd_outputs = self.net(foregnd_inputs)[:, self.target_cls]
        # backgnd_outputs = self.net(backgnd_inputs)[:, self.target_cls]

        # normalizing gradient
        losses = self.criterion((foregnd_outputs, backgnd_outputs), m)
        losses.backward(self.jacobian_vector_weights)

        alpha = 1 / torch.sqrt((m.grad.data**2).sum())
        m.grad.data.mul_(alpha)

    def _generate_noised_x(self, x):
        noise = torch.empty(self.batch_size, self.x_ch,
                            self.x_size, self.x_size).normal_(0, self.x_std)
        noise = noise.to(self.device)
        return x + noise

    def _step(self, m):
        self.optimizer.step()
        self.optimizer.zero_grad()
        m.data.clamp_(0, 1)

    def _predict(self, x):
        return F.softmax(self.net(x), 1)


class Loss(nn.Module):
    def __init__(self, lambda1, lambda2):
        super().__init__()

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eps = 1e-7  # to defense log(0)

    def forward(self, scores, m):
        foregnd_scores, backgnd_scores = scores
        foregnd_term = -torch.log(foregnd_scores)
        m_l1_term = self.lambda1 * torch.abs(m).view(m.size(0), -1).sum(dim=1)
        backgnd_term = -self.lambda2 * torch.log(1 - backgnd_scores + self.eps)
        return foregnd_term + m_l1_term + backgnd_term
