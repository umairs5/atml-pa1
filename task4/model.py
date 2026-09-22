from __future__ import annotations
import torch
from torch import nn
from torchvision.models import resnet18

class CifarResNet18(nn.Module):
    def __init__(self, classes: int = 10):
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity(); net.fc = nn.Linear(net.fc.in_features, classes)
        self.net = net
    def forward_features(self, x):
        n = self.net
        x = n.relu(n.bn1(n.conv1(x))); x = n.layer1(x); x = n.layer2(x); x = n.layer3(x); x = n.layer4(x)
        return n.avgpool(x).flatten(1)
    def forward(self, x): return self.net.fc(self.forward_features(x))
