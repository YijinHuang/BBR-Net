import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class MyModel(nn.Module):
    def __init__(self, backbone, bottleneck_size, num_classes, pretrained=False, **kwargs):
        super(MyModel, self).__init__()

        self.net = backbone(pretrained=pretrained, num_classes=1000, **kwargs)
        self.net.fc = nn.Sequential(
            nn.Linear(bottleneck_size, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        pred = self.net(x)
        return pred


class MyEfficientNet(nn.Module):
    def __init__(self, net_size, bottleneck_size, num_classes, pretrained=False):
        super(MyEfficientNet, self).__init__()

        if pretrained:
            self.net = EfficientNet.from_pretrained('efficientnet-b{}'.format(net_size))
        else:
            self.net = EfficientNet.from_name('efficientnet-b{}'.format(net_size))

        self.net._fc = nn.Sequential(
            nn.Linear(bottleneck_size, num_classes)
        )

    def forward(self, x):
        pred = self.net(x)
        return pred
