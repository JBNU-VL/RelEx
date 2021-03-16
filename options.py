import argparse
import torch


def _get_relex_opts(parent_parser):
    parser = argparse.ArgumentParser('RelEx', parents=[parent_parser])
    opts = parser.parse_args()

    # variabels
    opts.batch_size = 50
    opts.lr = 0.1
    opts.mtm = 0.99
    opts.lambda1 = 0.0001
    opts.lambda2 = 1.
    opts.x_std_level = 0.1
    opts.mode = 'batch'
    opts.max_iters = 50
    return opts


def _get_realtime_sal_opts(parent_parser):
    parser = argparse.ArgumentParser(
        'Real Time Saliency', parents=[parent_parser])
    opts = parser.parse_args()

    opts.model_confidence = 0.
    return opts


def _get_smoothgrad_opts(parent_parser):
    parser = argparse.ArgumentParser('SmoothGrad', parents=[parent_parser])
    opts = parser.parse_args()

    # variables
    opts.sample_size = 50
    opts.std_level = 0.1
    return opts


def _get_integrated_gradient_opts(parent_parser):
    parser = argparse.ArgumentParser(
        'Integrated Gradient', parents=[parent_parser])
    opts = parser.parse_args()

    # variables
    opts.steps = 100
    return opts


def _get_gradcam_opts(parent_parser):
    parser = argparse.ArgumentParser('GradCAM', parents=[parent_parser])
    opts = parser.parse_args()

    # variables
    opts.target_layers = 'layer4'
    opts.resize = True
    return opts


def _get_untargeted_opts(parent_parser):
    parser = argparse.ArgumentParser(
        'Projected Gradient Descent', parents=[parent_parser])
    opts = parser.parse_args()

    # variables
    opts.eps = 0.07
    opts.a = 0.01
    opts.K = 40
    import numpy as np
    opts.norm = np.inf
    max_bound = (1 - 0.406) / 0.225
    min_bound = (0 - 0.485) / 0.229
    opts.max_min_bounds = (max_bound, min_bound)

    # eps range in [0-mu/sigma, 1-mu/sigma]
    opts.eps_sets = (0.07, 0.1, 0.3, 1., 2., 4., 8.)
    return opts


def _get_structured_opts(parent_parser, parent_opts):
    parser = argparse.ArgumentParser(
        'Structured, ManipulationMethod', parents=[parent_parser])
    opts = parser.parse_args()

    # variables
    opts.lr = 0.0002
    opts.num_iters = 1500
    opts.factors = (1e11, 1e6)
    opts.beta_range = (10., 100.)

    # opts.x_max_min_bounds = ()

    # attack saliency methods
    opts.attack_methods = (
        'RealTimeSaliency', 'SmoothGrad', 'IntegratedGradient', 'GradCAM'
    )
    return opts


def _get_unstructured_opts(parent_parser):
    parser = argparse.ArgumentParser(
        'Unstructured, IterativeAttack', parents=[parent_parser])
    opts = parser.parse_args()

    # variables
    opts.eps = 1. / 255 / 0.225
    opts.k = 1000
    opts.num_iters = 100
    opts.alpha = 1
    opts.measurement = 'topk'
    opts.beta_growth = False
    opts.beta_range = (10., 100.)

    # eps range in [0, 255]
    opts.eps_sets = [1., 2., 4., 8.]
    # eps range in [0-mu/sigma, 1-mu/sigma]
    opts.scaled_eps_sets = (opts.eps_sets[0] / 255 / 0.225,
                            opts.eps_sets[1] / 255 / 0.225,
                            opts.eps_sets[2] / 255 / 0.225,
                            opts.eps_sets[3] / 255 / 0.225)

    # attack saliency methods
    opts.attack_methods = (
        'RealTimeSaliency', 'SmoothGrad', 'IntegratedGradient', 'GradCAM'
    )
    return opts


def get_opts():
    parser = argparse.ArgumentParser(
        'Building Reliable Explanations', add_help=False)
    parser.add_argument('--network', default='resnet50',
                        help='select between resnet and vgg, googlenet')
    parser.add_argument('--gpu', default=False, action='store_true',
                        help='flag value for usage of gpu')
    # assert function will be add..

    opts = parser.parse_args()
    opts.x_shape = (1, 3, 224, 224)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    mean_tensor = torch.as_tensor(mean).view(1, opts.x_shape[1], 1, 1)
    std_tensor = torch.as_tensor(std).view(1, opts.x_shape[1], 1, 1)
    opts.max_bound = (torch.ones(opts.x_shape) - mean_tensor) / std_tensor
    opts.min_bound = (torch.zeros(opts.x_shape) - mean_tensor) / std_tensor

    opts.x_ch = opts.x_shape[1]
    opts.x_size = opts.x_shape[2]

    # Adversarial
    opts.untargeted = _get_untargeted_opts(parser)
    opts.structured = _get_structured_opts(parser, opts)
    opts.unstructured = _get_unstructured_opts(parser)

    # Saliency
    opts.relex = _get_relex_opts(parser)
    opts.realtime_sal = _get_realtime_sal_opts(parser)
    opts.smoothgrad = _get_smoothgrad_opts(parser)
    opts.intgrad = _get_integrated_gradient_opts(parser)

    opts.available_adversarials = (
        'ProjectedGradientDescent', 'ManipulationMethod', 'IterativeAttack'
    )
    opts.available_saliencies = (
        'RelEx', 'RealTimeSaliency', 'SmoothGrad', 'IntegratedGradient',
        'GradCAM'
    )

    opts.gpus = '0,1,2,3,4'

    return opts


if __name__ == '__main__':
    opts = get_opts()
    print(opts.max_bound)
