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
    pass


def _get_integrated_gradient_opts():
    pass


def _get_untargeted_opts():
    pass


def _get_structured_opts():
    pass


def _get_unstructured_opts():
    pass


def get_opts():
    parser = argparse.ArgumentParser('Building Reliable Explanations')
    parser.add_argument('--network', default='resnet50',
                        help='select between resnet and vgg, googlenet')
    opts = parser.parse_args()
    opts.x_ch = 3
    opts.x_size = 224

    opts.relex = _get_relex_opts()
    return opts


if __name__ == '__main__':
    opts = get_opts()
