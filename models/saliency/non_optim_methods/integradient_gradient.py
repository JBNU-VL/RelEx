import torch
from torch import nn
from torch.autograd import grad


class IntegratedGradient(nn.Module):
    def __init__(self, net, steps=50, opts=None):
        super().__init__()

        self.net = net

        # variables
        self.steps = steps if steps != None else 50
        self.pathes = torch.linspace(
            0, 1, self.steps, dtype=torch.float32, device='cuda')

    def forward(self, x, target_cls, sec_ord=False):
        x.requires_grad_(True)
        pathes_x = self.pathes.view(self.steps, 1, 1, 1) * x
        outputs = self.net(pathes_x)
        sal = grad(outputs.mean(), x, create_graph=sec_ord)[0] * x.detach()
        return sal


if __name__ == '__main__':
    import torchvision
    net = torchvision.models.resnet50(True).to(0)
    IG_attr = IntegratedGradient(net)
    x = torch.rand(1, 3, 224, 224).to(0)
    ig = IG_attr(x, 0)
    print(ig.size())
