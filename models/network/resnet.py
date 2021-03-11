import torch
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, Bottleneck

'''
DeepLift: relu position is changed
RT-Sal: ResNet50encoder
'''


class ResNetEncoder(ResNet):
    def forward(self, x):
        s0 = x
        x = self.conv1(s0)
        x = self.bn1(x)
        s1 = self.relu(x)
        x = self.maxpool(s1)

        s2 = self.layer1(x)
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)

        x = self.avgpool(s5)
        sX = x.view(x.size(0), -1)
        sC = self.fc(sX)

        return s0, s1, s2, s3, s4, s5, sX, sC


def resnet50(encoder=False, robust=True, **kwargs):
    if encoder:
        net = ResNetEncoder(Bottleneck, [3, 4, 6, 3], **kwargs)
    else:
        net = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if not robust:
        net.load_state_dict(model_zoo.load_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    else:
        pass

    return net


if __name__ == '__main__':
    import torchvision
    net1 = torchvision.models.resnet50(True).eval()
    net2 = resnet50(encoder=False, pretrained=True).eval()
    x = torch.rand(1, 3, 224, 224)

    with torch.no_grad():
        outputs1 = net1(x)
        outputs2 = net2(x)

        max_cls = outputs1.max(1)[1]
        print(
            f'outputs1: {outputs1[0,max_cls]}, outputs2: {outputs2[0,max_cls]}')
