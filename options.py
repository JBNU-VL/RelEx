import argparse


def _get_relex_opts():
    parser = argparse.ArgumentParser('RelEx')
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


def _get_realtime_sal_opts():
    parser = argparse.ArgumentParser('Real Time Saliency')
    opts = parser.parse_args()

    opts.model_confidence = 0.
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


def _get_gradcam_opts():
    parser = argparse.ArgumentParser('GradCAM')
    opts = parser.parse_args()

    # variables
    opts.target_layers = 'layer4'
    opts.resize = True
    return opts


def _get_untargeted_opts():
    parser = argparse.ArgumentParser('Projected Gradient Descent')
    opts = parser.parse_args()

    # variables
    opts.eps = 0.07
    opts.a = 0.01
    opts.K = 40
    opts.max_bound = (1 - 0.406) / 0.225
    opts.min_bound = (0 - 0.485) / 0.229

    # eps range in [0-mu/sigma, 1-mu/sigma]
    opts.eps_sets = (0.07, 0.1, 0.3, 1., 2., 4., 8.)
    return opts


def _get_structured_opts():
    parser = argparse.ArgumentParser('Structured, ManipulationMethod')
    opts = parser.parse_args()

    # variables
    opts.lr = 0.0002
    opts.num_iters = 1500
    opts.factors = (1e11, 1e6)
    opts.beta_range = (10., 100.)

    # attack saliency methods
    opts.attack_methods = (
        'RealTimeSaliency', 'SmoothGrad', 'IntegratedGradient', 'GradCAM'
    )
    return opts


def _get_unstructured_opts():
    parser = argparse.ArgumentParser('Unstructured, IterativeAttack')
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
    eps_sets = [1., 2., 4., 8.]
    # eps range in [0-mu/sigma, 1-mu/sigma]
    opts.eps_sets = (eps_sets[0] / 255 / 0.225,
                     eps_sets[1] / 255 / 0.225,
                     eps_sets[2] / 255 / 0.225,
                     eps_sets[3] / 255 / 0.225)

    # attack saliency methods
    opts.attack_methods = (
        'RealTimeSaliency', 'SmoothGrad', 'IntegratedGradient', 'GradCAM'
    )
    return opts


def get_opts():
    parser = argparse.ArgumentParser('Building Reliable Explanations')
    parser.add_argument('--network', default='resnet50',
                        help='select between resnet and vgg, googlenet')
    # assert function will be add..

    opts = parser.parse_args()

    opts.x_ch = 3
    opts.x_size = 224

    # Adversarial
    opts.untargeted = _get_untargeted_opts()
    opts.structured = _get_structured_opts()
    opts.unstructured = _get_unstructured_opts()

    # Saliency
    opts.relex = _get_relex_opts()
    opts.realtime_sal = _get_realtime_sal_opts()
    opts.smoothgrad = _get_smoothgrad_opts()
    opts.intgrad = _get_integrated_gradient_opts()

    opts.available_adversarials = (
        'ProjectedGradientDescent', 'ManipulationMethod', 'IterativeAttack'
    )
    opts.available_saliencies = (
        'RelEx', 'RealTimeSaliency', 'SmoothGrad', 'IntegratedGradient',
        'GradCAM'
    )

    return opts


if __name__ == '__main__':
    opts = get_opts()
