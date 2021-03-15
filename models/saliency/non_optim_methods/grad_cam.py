import torch
from torch import nn
import torch.nn.functional as F


class GradCAM(nn.Module):
    def __init__(self, net, target_layers, resize=True):
        super().__init__()
        self.net = net
        if isinstance(target_layers, torch.nn.Module):
            target_layers = [target_layers]
        self.target_layers = target_layers
        self.resize = resize

    @torch.enable_grad()
    def forward(self, x, target_cls=None, sec_ord=False):
        if not sec_ord:
            x.requires_grad_(True)

        self.net.zero_grad()
        features, handlers = self.register()

        grad_cams = self._make_cam(x, target_cls, features, sec_ord)

        for handler in handlers:
            handler.remove()

        if self.resize:
            _gc = []
            for grad_cam in grad_cams:
                _gc.append(
                    F.interpolate(
                        grad_cam, size=(x.size(-2), x.size(-1)),
                        mode='bilinear', align_corners=True
                    )
                )
            grad_cams = _gc

        accu = self.predict(x)
        if len(grad_cams) == 1:
            return grad_cams[0], accu
        return grad_cams, accu

    def register(self):
        handlers = []
        features = []

        def forward_hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                features.append(outputs[0])
            else:
                features.append(outputs)

        for target_layer in self.target_layers:
            handlers.append(
                target_layer.register_forward_hook(forward_hook))

        return features, handlers

    def predict(self, x):
        return F.softmax(self.net(x), 1)

    def to_onehot(self, indices, num_classes):
        onehot = torch.zeros(indices.shape[0], num_classes, *indices.shape[1:],
                             dtype=torch.float,
                             device=indices.device)
        onehot.scatter_(1, indices.unsqueeze(1), 1)
        return onehot

    # @torch.no_grad()
    def _make_cam(self, inputs, target, features, sec_ord=False):
        if target is None:
            output = self.net(inputs)
            target = output.max(1)[1]
        else:
            output = self.net(inputs)

        onehot = self.to_onehot(target, output.shape[-1])
        loss = (output * onehot).sum()

        grad_cams = []
        for feature in features:
            # print('feature', feature.requires_grad)
            grad = torch.autograd.grad(
                loss, feature, create_graph=False, retain_graph=True
            )[0]
            weight = F.adaptive_avg_pool2d(grad, 1)
            grad_cam = feature.mul(weight)
            grad_cam = F.relu(
                grad_cam.sum(dim=1, keepdim=True))
            grad_cams.append(grad_cam)
        return grad_cams
