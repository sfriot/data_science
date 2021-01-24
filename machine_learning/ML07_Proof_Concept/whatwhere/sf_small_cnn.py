# -*- coding: utf-8 -*-
"""
Created on Tue Nov 5 2020

@author: Sylvain Friot
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    import sf_features_extraction as sffe
else:
    import modules_perso.sf_features_extraction as sffe


__all__ = ["MyCNN", "mycnn3"]
model_mycnn3_url = "not_trained.pth" # pre-trained data -> error


class MyCNN(sffe.FeatureExtraction):

    def __init__(self, name, in_channels, nb_filters_first_conv, nb_conv,
                 include_dropout=True, num_classes=1000,
                 max_num_features=None, folder_saves=None, init_weights=True):
        self.model_name = name
        self.model_family = "mycnn"
        super().__init__(max_num_features, folder_saves)
        self.num_classes = num_classes

        layers = []
        filters_in = in_channels
        filters_out = nb_filters_first_conv
        for i in range(nb_conv):
            layers.append(nn.Conv2d(filters_in, filters_out, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(filters_out))
            layers.append(nn.ReLU())
            if include_dropout:
                layers.append(nn.Dropout2d(p=0.2))
            layers.append(nn.MaxPool2d(2))
            filters_in = filters_out
            filters_out = filters_out * 2
        self.convblocks = nn.Sequential(*layers)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(filters_in, self.num_classes)
        
        if init_weights:
            self._initialize_weights()
        self.get_num_features()

    def forward(self, x, include_feat=False):
        feat = []
        previous_x = self.convblocks[0](x)
        for layer in self.convblocks:
            x = layer(x)
            if x.size(2) < previous_x.size(2):  # reduction level, previous_x is a layer before reduction
                feat.append(previous_x)
            previous_x = x

        out = self.maxpool(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        if include_feat:
            return out, feat
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def mycnn3(num_classes=1000, include_dropout=True, pretrained=False, in_channels=3,
           max_num_features=None, folder_saves=None,
           url_pretrained=None, **kwargs):
    model = MyCNN(name="mycnn3", in_channels=in_channels, nb_filters_first_conv=32,
                  nb_conv=3, include_dropout=include_dropout, num_classes=num_classes,
                  max_num_features=max_num_features, folder_saves=folder_saves,
                  **kwargs)
    if pretrained:
        if url_pretrained is None:
            model.load_model(model_mycnn3_url)
        else:
            model.load_model(url_pretrained)
    return model
