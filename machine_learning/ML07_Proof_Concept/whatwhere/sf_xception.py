# -*- coding: utf-8 -*-
"""
Created on Tue Nov 3 2020

@author: Sylvain Friot
Based on https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py
"""


from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    import sf_features_extraction as sffe
else:
    import modules_perso.sf_features_extraction as sffe


__all__ = ['xception']

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1,
                                   bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps,
                 strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1,
                                  stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        rep = []
        filters = in_filters
        if grow_first:
            if start_with_relu:
                rep.append(nn.ReLU(inplace=False))
            rep.append(SeparableConv2d(in_filters, out_filters, 3,
                                       stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3,
                                       stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3,
                                       stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(sffe.FeatureExtraction):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, name, num_classes=1000,
                 max_num_features=None, folder_saves=None, init_weights=True):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        self.model_name = name
        self.model_family = "xception"
        super().__init__(max_num_features, folder_saves)
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("conv1", nn.Conv2d(3, 32, 3, 2, 0, bias=False))
        self.features.add_module("bn1", nn.BatchNorm2d(32))
        self.features.add_module("relu1", nn.ReLU(inplace=True))
        self.features.add_module("conv2", nn.Conv2d(32, 64, 3, bias=False))
        self.features.add_module("bn2", nn.BatchNorm2d(64))
        self.features.add_module("relu2", nn.ReLU(inplace=True))
        self.features.add_module("block1", 
                                 Block(64, 128, 2, 2, start_with_relu=False, grow_first=True))
        self.features.add_module("block2",
                                 Block(128, 256, 2, 2, start_with_relu=True, grow_first=True))
        self.features.add_module("block3",
                                 Block(256, 728, 2, 2, start_with_relu=True, grow_first=True))
        self.features.add_module("block4",
                                 Block(728, 728, 3, 1, start_with_relu=True, grow_first=True))
        self.features.add_module("block5",
                                 Block(728, 728, 3, 1, start_with_relu=True, grow_first=True))
        self.features.add_module("block6",
                                 Block(728, 728, 3, 1, start_with_relu=True, grow_first=True))
        self.features.add_module("block7",
                                 Block(728, 728, 3, 1, start_with_relu=True, grow_first=True))
        self.features.add_module("block8",
                                 Block(728, 728, 3, 1, start_with_relu=True, grow_first=True))
        self.features.add_module("block9",
                                 Block(728, 728, 3, 1, start_with_relu=True, grow_first=True))
        self.features.add_module("block10",
                                 Block(728, 728, 3, 1, start_with_relu=True, grow_first=True))
        self.features.add_module("block11",
                                 Block(728, 728, 3, 1, start_with_relu=True, grow_first=True))
        self.features.add_module("block12",
                                 Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False))
        self.features.add_module("conv3", SeparableConv2d(1024, 1536, 3, 1, 1))
        self.features.add_module("bn3", nn.BatchNorm2d(1536))
        self.features.add_module("relu3", nn.ReLU(inplace=True))
        self.features.add_module("conv4", SeparableConv2d(1536, 2048, 3, 1, 1))
        self.features.add_module("bn4", nn.BatchNorm2d(2048))
        self.features.add_module("relu4", nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, self.num_classes)

        if init_weights:
            self._initialize_weights()
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
        feat.append(x)

        out = self.avgpool(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        if include_feat:
            return out, feat
        return out

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


def xception(num_classes=1000, pretrained=False,
             max_num_features=None, folder_saves=None, **kwargs):
    model = Xception("xception", num_classes=num_classes,
                     max_num_features=max_num_features, folder_saves=folder_saves,
                     **kwargs)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}"\
            .format(settings['num_classes'], num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

    return model
