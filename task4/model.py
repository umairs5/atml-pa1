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
        x = self.forward_from_layer2(self.forward_to_layer2(x))
        return n.avgpool(x).flatten(1)
    def forward_to_layer2(self, x):
        n=self.net; return n.layer2(n.layer1(n.relu(n.bn1(n.conv1(x)))))
    def forward_from_layer2(self, x): return self.net.layer4(self.net.layer3(x))
    def forward(self, x): return self.net.fc(self.forward_features(x))
