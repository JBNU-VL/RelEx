import os
import torch
from torch import nn
import torch.nn.functional as F

from ._modules.realtime_saliency import SaliencyModel


class RealTimeSaliency(nn.Module):
    def __init__(self, net, model_confidence=0, device=None):
        super().__init__()
        model = SaliencyModel(net, 5, 64, 3, 64, fix_encoder=True,
                              use_simple_activation=False, allow_selector=True)
        model.minimialistic_restore(os.path.join(
            os.path.dirname(__file__), '_modules', 'realtime_saliency',
            'minsaliency'))
        model.train(False)
        self.model = model.to(device)

        self.model_confidence = model_confidence

    def forward(self, x, target_cls):
        masks, _, cls_logits = self.model(
            x * 2, target_cls, model_confidence=self.model_confidence)
        sal_map = F.interpolate(masks, x.size(
            2), mode='bilinear', align_corners=False)
        return sal_map, cls_logits
