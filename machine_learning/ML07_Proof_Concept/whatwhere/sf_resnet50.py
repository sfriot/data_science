# -*- coding: utf-8 -*-
"""
Created on Tue Nov 3 2020

@author: Sylvain Friot
Based on https://github.com/alinlab/L2T-ww/blob/master/models/resnet_ilsvrc.py
"""

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    import sf_features_extraction as sffe
else:
    import modules_perso.sf_features_extraction as sffe


__all__ = ["ResNet", "resnet50", "resnet18", "ResNet_v2", "resnet50_v2"]

# url de resnet50 prÃ©-entraÃ®nÃ©
model_resnet50_url = \
    "https://download.pytorch.org/models/resnet50-19c8e357.pth"
model_resnet18_url = \
    "https://download.pytorch.org/models/resnet18-5c106cde.pth"


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class ResNet(sffe.FeatureExtraction):

    def __init__(self, name, block, layers, num_classes=1000,
                 max_num_features=None, folder_saves=None, init_weights=True):
        self.model_name = name
        self.model_family = "resnet"
        super().__init__(max_num_features, folder_saves)
        self.num_classes = num_classes
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)
        self.lwf = False
        if init_weights:
            self._initialize_weights()
        self.get_num_features()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, include_feat=False):
        f1 = self.conv1(x)
        b1 = self.bn1(f1)
        r1 = self.relu(b1)
        p1 = self.maxpool(r1)

        f2 = self.layer1(p1)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)

        f6 = self.avgpool(f5)
        f6 = f6.view(f6.size(0), -1)
        f7 = self.fc(f6)

        if include_feat:
            return f7, [r1, f2, f3, f4, f5]
        return f7


def resnet50(num_classes=1000, pretrained=False,
             max_num_features=None, folder_saves=None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet("resnet50", Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
                   max_num_features=max_num_features, folder_saves=folder_saves, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_resnet50_url)
        if num_classes != 1000:
            state_dict.pop("fc.weight")
            state_dict.pop("fc.bias")
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(num_classes=1000, pretrained=False,
             max_num_features=None, folder_saves=None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet("resnet18", BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
                   max_num_features=max_num_features, folder_saves=folder_saves, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_resnet18_url)
        if num_classes != 1000:
            state_dict.pop("fc.weight")
            state_dict.pop("fc.bias")
        model.load_state_dict(state_dict, strict=False)
    return model


class Bottleneck_v2(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_v2, self).__init__()
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        else:
            residual = x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class ResNet_v2(sffe.FeatureExtraction):

    def __init__(self, name, block, layers, num_classes=1000,
                 max_num_features=None, folder_saves=None):
        self.model_name = name
        self.model_family = "resnet"
        super().__init__(max_num_features, folder_saves)
        self.num_classes = num_classes
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)
        self.lwf = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.get_num_features()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, include_feat=False):
        f1 = self.conv1(x)
        f2 = self.layer1(f1)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        f6 = self.avgpool(f5)
        f6 = f6.view(f6.size(0), -1)
        f7 = self.fc(f6)
        if include_feat:
            return f7, [f1, f2, f3, f4, f5]
        return f7


def resnet50_v2(num_classes=1000, pretrained=False,
                max_num_features=None, folder_saves=None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_v2("resnet50_v2", Bottleneck_v2, [3, 4, 6, 3], num_classes=num_classes,
                      max_num_features=max_num_features, folder_saves=folder_saves, **kwargs)
    if pretrained:
        print("Pre Trained not available with Pytorch")
    return model
