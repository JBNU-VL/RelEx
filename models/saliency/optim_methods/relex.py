import torch
from torch import nn
from torch.autograd import grad

r'''
dd
'''


class RelEx(nn.Module):
    def __init__(self, net, opts):
        # common variables
        self.x_ch = opts.x_ch
        self.x_size = opts.x_size

        # hyper-parameters for RelEx
        self.batch_size = opts.relex.batch_size  # 50
        self.lr = opts.relex.lr  # 0.1
        self.mtm = opts.relex.mtm  # 0.99

        # RelEx variables
        self.mode = opts.relex.mode  # 'batch' or 'vanilla'
        self.max_iters = opts.relex.max_iters  # 50 ~ 100

        self.jacobian_vector_ones = torch.ones(
            self.batch_size, dtype=torch.float32, device='cuda')
        self.jacobian_vector_weights = torch.ones(
            self.batch_size, dtype=torch.float32, device='cuda') / self.batch_size

        self.net = net
        self.criterion = Loss(opts.relex.lambda1, opts.relex.lambda2)

    def forward(self, x, target_cls, sec_ord=False):
        m = self._reset(x, sec_ord)

        m_sets = []
        for i in range(self.max_iters):
            self._generate_grad(x, m)
            self._step(m)
            if i == 0 or (i+1) == self.max_iters or (i+1) % self.save_freq == 0:
                m_sets.append(m.detach().clone())

        return m_sets

    def _reset(self, x, sec_ord):
        self.sec_ord = sec_ord  # whether calculate second order derivative

        self.x_std = (x.max() - x.min()) * self.x_std_level  # image std

        # initialize mask
        m = torch.rand(1, self.x_ch, self.x_size, self.x_size) / 100
        m.requires_grad_(True)
        self.optimizer = torch.optim.SGD([m], lr=self.lr, momentum=self.mtm)
        return m

    def _generate_grad(self, x, m):
        batch_x = self._generate_noised_x(x)

        foregnd_inputs = batch_x * m
        backgnd_inputs = batch_x * (1 - m)

        foregnd_outputs = self.net(foregnd_inputs)
        backgnd_outputs = self.net(backgnd_inputs)

        # normalizing gradient
        losses = self.criterion((foregnd_outputs, backgnd_outputs), m)
        # m_grad = grad(losses.mean(), m, create_graph=self.sec_ord)[0]
        losses.backward(self.jacobian_vector_weights)
        alpha = 1 / torch.sqrt((m.grad.data**2).sum())
        m.grad.data.mul_(alpha)

    def _generate_noised_x(self, x):
        noise = torch.empty(self.batch_size, self.x_ch,
                            self.x_size).normal_(0, self.x_std)
        if torch.cuda.is_available():
            noise = noise.to(0)
        return x + noise

    def _step(self, m):
        self.optimizer.step()
        self.optimizer.zero_grad()


class Loss(nn.Module):
    def __init__(self, lambda1, lambda2):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eps = 1e-7  # to defense log(0)

    def forward(self, scores, m):
        foregnd_scores, backgnd_scores = scores
        foregnd_term = -torch.log(foregnd_scores)
        m_l1_term = self.lambda1 * torch.abs(m).view(m.size(0), -1).sum(dim=1)
        backgnd_term = -self.lambda2 * torch.log(1 - backgnd_scores + self.eps)
        return foregnd_term + m_l1_term + backgnd_term
