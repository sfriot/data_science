# -*- coding: utf-8 -*-
"""
Created on Tue Nov 3 2020

@author: Sylvain Friot
Based on https://github.com/alinlab/L2T-ww/blob/master/models/vgg_cifar.py
"""


import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    import sf_features_extraction as sffe
else:
    import modules_perso.sf_features_extraction as sffe


__all__ = ["VGG", "vgg16"]


# url de vgg16 pré-entrainé
model_vgg16_url = \
    "https://download.pytorch.org/models/vgg16-397923af.pth"
model_vgg16bn_url = \
    "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"


class VGG(sffe.FeatureExtraction):

    def __init__(self, name, features, num_classes=1000, original_classifier=True,
                 max_num_features=None, folder_saves=None, init_weights=True):
        self.model_name = name
        self.model_family = "vgg"
        super().__init__(max_num_features, folder_saves)
        self.num_classes = num_classes
        self.features = features
        if original_classifier:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes))
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(512, self.num_classes)
        if init_weights:
            self._initialize_weights()
        self.get_num_features()

    def forward(self, x, include_feat=False):
        feat = []
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                feat.append(x)
            x = layer(x)
        out = self.avgpool(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        if include_feat:
            return out, feat
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def freeze_layers(self, first_layer=0, last_layer=None):
        if last_layer is None:
            last_layer = len(list(self.children()))
        if last_layer < 0:
            last_layer += len(list(self.children()))
        idx_layer = 0
        for name, child in self.named_children():
            print("{} : {}".format(idx_layer, name))
            if (idx_layer >= first_layer) & (idx_layer < last_layer):
                print("freezed")
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print("unfreezed")
                for param in child.parameters():
                    param.requires_grad = True
            idx_layer += 1

    def unfreeze_layers(self, first_layer=0, last_layer=None):
        if last_layer is None:
            last_layer = len(list(self.children()))
        if last_layer < 0:
            last_layer += len(list(self.children()))
        idx_layer = 0
        for name, child in self.named_children():
            print("{} : {}".format(idx_layer, name))
            if (idx_layer >= first_layer) & (idx_layer < last_layer):
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False
            idx_layer += 1


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(num_classes=1000, original_classifier=True, pretrained=False,
          max_num_features=None, folder_saves=None, **kwargs):
    """
    VGG 16-layer model
    """
    cfg_vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                 512, 512, 512, 'M', 512, 512, 512, 'M']
    model = VGG("vgg16", make_layers(cfg_vgg16), num_classes=num_classes,
                original_classifier=original_classifier,
                max_num_features=max_num_features, folder_saves=folder_saves,
                **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_vgg16_url)
        if original_classifier & (num_classes == 1000):
            model.load_state_dict(state_dict)
        else:
            if not original_classifier:
                state_dict.pop("classifier.0.weight")
                state_dict.pop("classifier.0.bias")
                state_dict.pop("classifier.3.weight")
                state_dict.pop("classifier.3.bias")
            state_dict.pop("classifier.6.weight")
            state_dict.pop("classifier.6.bias")
            model.load_state_dict(state_dict, strict=False)
    return model


def vgg16bn(num_classes=1000, original_classifier=True, pretrained=False,
            max_num_features=None, folder_saves=None, **kwargs):
    """
    VGG 16-layer model with batch normalization
    """
    cfg_vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                 512, 512, 512, 'M', 512, 512, 512, 'M']
    model = VGG("vgg16", make_layers(cfg_vgg16, batch_norm=True), num_classes=num_classes,
                original_classifier=original_classifier,
                max_num_features=max_num_features, folder_saves=folder_saves,
                **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_vgg16bn_url)
        if original_classifier & (num_classes == 1000):
            model.load_state_dict(state_dict)
        else:
            if not original_classifier:
                state_dict.pop("classifier.0.weight")
                state_dict.pop("classifier.0.bias")
                state_dict.pop("classifier.3.weight")
                state_dict.pop("classifier.3.bias")
            state_dict.pop("classifier.6.weight")
            state_dict.pop("classifier.6.bias")
            model.load_state_dict(state_dict, strict=False)
    return model
