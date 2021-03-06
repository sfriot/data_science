# -*- coding: utf-8 -*-
"""
Created on Tue Nov 3 2020

@author: Sylvain Friot
Based on https://pytorch.org/docs/stable/_modules/torchvision/models/densenet.html#densenet201
J'ai retirÃ© l'option memory efficient pour un code plus concis
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torch import Tensor

import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    import sf_features_extraction as sffe
else:
    import modules_perso.sf_features_extraction as sffe


__all__ = ['DenseNet', 'densenet201', 'densenet121']

model_densenet201_url = \
    "https://download.pytorch.org/models/densenet201-c1103571.pth"
model_densenet121_url = \
    "https://download.pytorch.org/models/densenet121-a639ec97.pth"


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features,
                                           bn_size * growth_rate, kernel_size=1,
                                           stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input
        concated_features = torch.cat(prev_features, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size,
                 growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate=growth_rate, bn_size=bn_size,
                                drop_rate=drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features,
                                          num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(sffe.FeatureExtraction):
    """
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, name, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=1000, max_num_features=None, folder_saves=None):
        self.model_name = name
        self.model_family = "densenet"
        super().__init__(max_num_features, folder_saves)
        self.num_classes = num_classes
        
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, self.num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.get_num_features()

    def forward(self, x, include_feat=False):
        feat = []
        x = self.features[0](x)
        previous_x = x

        for layer in self.features[1:]:
            x = layer(x)
            if x.size(2) < previous_x.size(2):  # reduction level, previous_x is a layer before reduction
                feat.append(previous_x)
            previous_x = x
        x = F.relu(x, inplace=True)
        feat.append(x)

        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        if include_feat:
            return out, feat
        return out


def densenet201(num_classes=1000, pretrained=False,
                max_num_features=None, folder_saves=None, **kwargs):
    model = DenseNet("densenet201", growth_rate=32, block_config=(6, 12, 48, 32),
                     num_init_features=64, num_classes=num_classes,
                     max_num_features=max_num_features, folder_saves=folder_saves,
                     **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_densenet201_url)
        if num_classes != 1000:
            state_dict.pop("classifier.weight")
            state_dict.pop("classifier.bias")
        temp_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("norm.1", "norm1")
            new_key = new_key.replace("norm.2", "norm2")
            new_key = new_key.replace("conv.1", "conv1")
            new_key = new_key.replace("conv.2", "conv2")
            temp_dict[new_key] = v
        model.load_state_dict(temp_dict, strict=False)
    return model


def densenet121(num_classes=1000, pretrained=False,
                max_num_features=None, folder_saves=None, **kwargs):
    model = DenseNet("densenet121", growth_rate=32, block_config=(6, 12, 24, 16),
                     num_init_features=64, num_classes=num_classes,
                     max_num_features=max_num_features, folder_saves=folder_saves,
                     **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_densenet121_url)
        if num_classes != 1000:
            state_dict.pop("classifier.weight")
            state_dict.pop("classifier.bias")
        temp_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("norm.1", "norm1")
            new_key = new_key.replace("norm.2", "norm2")
            new_key = new_key.replace("conv.1", "conv1")
            new_key = new_key.replace("conv.2", "conv2")
            temp_dict[new_key] = v
        model.load_state_dict(temp_dict, strict=False)
    return model
