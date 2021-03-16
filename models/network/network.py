import torch
import torch.utils.model_zoo as model_zoo

from .resnet import resnet50


def load_network(network_name, encoder=False, robust=False, freezing=True, **kwargs):
    if network_name == 'resnet50':
        net = resnet50(encoder, robust)

    load_weight(net, network_name, robust)

    if freezing:
        for p in net.parameters():
            p.requires_grad_(False)

    return net.eval()


def load_weight(net, network_name, robust=False):
    if network_name == 'resnet50':
        if robust:
            import os
            weight_full_dir = os.path.join(os.path.dirname(__file__),
                                           'checkpoint/robust_resnet50.pt')
            weight = torch.load(weight_full_dir)
        else:
            weight = model_zoo.load_url(
                'https://download.pytorch.org/models/resnet50-19c8e357.pth')

    net.load_state_dict(weight)
