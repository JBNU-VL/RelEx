import os
import torch
from torch import nn
import torch.nn.functional as F

from ._modules.realtime_saliency import SaliencyModel


class RealTimeSaliency(nn.Module):
    def __init__(self, net, model_confidence=0, device=None):
        super().__init__()
        import copy
        self.net = copy.deepcopy(net).to(device)

        model = SaliencyModel(self.net, 5, 64, 3, 64, fix_encoder=True,
                              use_simple_activation=False, allow_selector=True)
        model.minimialistic_restore(os.path.join(
            os.path.dirname(__file__), '_modules', 'realtime_saliency',
            'minsaliency'))
        model.train(False)
        self.model = model.to(device)

        self.model_confidence = model_confidence

    def forward(self, x, target_cls=None, sec_ord=False):
        if target_cls == None:
            target_cls = self.net(x.detach())[-1].max(1)[1]

        masks = self.model(x * 2, target_cls,
                           model_confidence=self.model_confidence)[0]
        sal = F.interpolate(masks, x.size(
            2), mode='bilinear', align_corners=False)
        accu = self._predict(x)
        return sal, accu

    def _predict(self, x):
        return F.softmax(self.net(x)[-1], 1)
