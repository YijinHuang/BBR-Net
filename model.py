import torch
import torch.nn as nn
import torchvision.models as models


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