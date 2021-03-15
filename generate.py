import torch

from .models.adversarial import pgd_fn, ManipulationMethod, IterativeAttack
from .models.saliency import RelEx, RealTimeSaliency, GradCAM, SmoothGrad, IntegratedGradient
from .models.network import resnet50
from .options import get_opts


class SaliencyGenerator:
    def __init__(self, opts, expl):
        pass

    def __call__(self, x):
        return x.sum()


if __name__ == '__main__':
    opts = get_opts()
