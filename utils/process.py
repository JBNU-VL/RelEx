import torch
import torchvision
import numpy as np


def preprocess_image(img, size=224, center_crop=True):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if center_crop:
        x = torchvision.transforms.Normalize(mean=mean, std=std)(
            torchvision.transforms.ToTensor()(
                torchvision.transforms.CenterCrop(size)(
                    torchvision.transforms.Resize(256)(img))))
        x0 = torchvision.transforms.ToTensor()(
            torchvision.transforms.CenterCrop(size)(
                torchvision.transforms.Resize(256)(img)))
    else:
        x = torchvision.transforms.Normalize(mean=mean, std=std)(
            torchvision.transforms.ToTensor()(
                torchvision.transforms.Resize((size, size))(img)))
        x0 = torchvision.transforms.ToTensor()(
            torchvision.transforms.Resize((size, size))(img))

    return x, x0


def depreprocess_image():
    pass


def normalize(sal, plane=True, percentile=True):
    sal = torch.abs(sal)[0]
    if plane:
        sal = sal.sum(dim=0)

    if percentile:
        sal_max = np.percentile(sal, 99)
    else:
        sal_max = sal.max()

    sal_min = sal.min()
    return torch.clamp((sal - sal_min) / (sal_max - sal_min), min=0, max=1)
