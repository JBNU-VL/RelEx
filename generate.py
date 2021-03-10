import torch


class SaliencyGenerator:
    def __init__(self, opts, expl):
        pass

    def __call__(self, x):
        return x.sum()


if __name__ == '__main__':
    sal_generator = SaliencyGenerator()
    x = torch.rand(1, 3, 224, 224)
    print(exp_generator(x))
