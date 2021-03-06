import torch
import torchvision
from torchvision.models.resnet import ResNet

'''
DeepLift: relu position is changed
RT-Sal: ResNet50encoder
'''


class Bottleneck(torchvision.models.resnet.Bottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.relu
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.relu3 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


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


def resnet50(encoder=False, robust=False, **kwargs):
    if encoder and not robust:
        net = ResNetEncoder(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif not encoder and not robust:
        net = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif not encoder and robust:
        net = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10, **kwargs)
    else:
        net = None

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
