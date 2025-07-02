
"""
Implementação do modelo SE-ResNet1D para classificação de ECG
"""

import torch
import torch.nn as nn

from .base_model import BaseModel
from ..config.model_configs import SEResNet1DConfig


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=15, stride=stride, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=15, stride=1, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SEBlock(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEResNet1D(BaseModel):
    """SE-ResNet1D para classificação de ECG"""
    
    def __init__(
        self,
        config: SEResNet1DConfig,
        num_classes: int = 5,
        input_channels: int = 12,
        **kwargs
    ):
        super().__init__(num_classes, input_channels)
        self.config = config
        
        self.inplanes = 64
        
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=config.kernel_size, stride=2, padding=config.kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock1D, config.channels[0], config.layers[0], stride=1, reduction=config.reduction)
        self.layer2 = self._make_layer(BasicBlock1D, config.channels[1], config.layers[1], stride=2, reduction=config.reduction)
        self.layer3 = self._make_layer(BasicBlock1D, config.channels[2], config.layers[2], stride=2, reduction=config.reduction)
        self.layer4 = self._make_layer(BasicBlock1D, config.channels[3], config.layers[3], stride=2, reduction=config.reduction)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(config.channels[-1] * BasicBlock1D.expansion, num_classes)
        
        self.dropout = nn.Dropout(config.dropout)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, reduction=16):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, reduction))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, reduction=reduction))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
