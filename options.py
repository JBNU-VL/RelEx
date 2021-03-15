import argparse


def _get_relex_opts():
    parser = argparse.ArgumentParser('RelEx')
    opts = parser.parse_args()

    # hyper-parameters
    opts.batch_size = 50
    opts.lr = 0.1
    opts.mtm = 0.99
    opts.lambda1 = 0.0001
    opts.lambda2 = 1.
    opts.x_std_level = 0.1

    # variabels
    opts.mode = 'batch'
    opts.max_iters = 50
    return opts


def _get_smoothgrad_opts():
    parser = argparse.ArgumentParser('SmoothGrad')
    opts = parser.parse_args()

    # variables
    opts.sample_size = 50
    opts.std_level = 0.1
    return opts


def _get_integrated_gradient_opts():
    parser = argparse.ArgumentParser('Integrated Gradient')
    opts = parser.parse_args()

    # variables
    opts.steps = 100
    return opts


def _get_untargeted_opts():
    parser = argparse.ArgumentParser('Projected Gradient Descent')
    opts = parser.parse_args()

    opts.eps = 0.07
    opts.a = 0.01
    opts.K = 40
    opts.max_val = (1 - 0.406) / 0.225
    opts.min_val = (0 - 0.485) / 0.229)
    return opts

def _get_structured_opts():
    parser=argparse.ArgumentParser('Structured, ManipulationMethod')
    opts=parser.parse_args()

    opts.lr=0.0002
    opts.num_iters=1500
    opts.factors=(1e11, 1e6)
    opts.beta_range=(10., 100.)
    return opts


def _get_unstructured_opts():
    parser=argparse.ArgumentParser('Unstructured, IterativeAttack')
    opts=parser.parse_args()

    opts.eps=1. / 255 / 0.225
    opts.k=1000
    opts.num_iters=100
    opts.alpha=1
    opts.measurement='topk'
    opts.beta_growth=False
    opts.beta_range=(10., 100.)
    return opts


def get_opts():
    parser=argparse.ArgumentParser('Building Reliable Explanations')
    parser.add_argument('--network', default = 'resnet50',
                        help = 'select between resnet and vgg, googlenet')
    # assert function will be add..

    opts=parser.parse_args()

    opts.x_ch=3
    opts.x_size=224

    opts.relex=_get_relex_opts()
    opts.smoothgrad=_get_smoothgrad_opts()
    opts.integrated_gradient=_get_integrated_gradient_opts()
    opts.untargeted=_get_untargeted_opts()
    opts.structured=_get_structured_opts()
    opts.unstructured=_get_unstructured_opts()
    return opts


if __name__ == '__main__':
    opts=get_opts()
