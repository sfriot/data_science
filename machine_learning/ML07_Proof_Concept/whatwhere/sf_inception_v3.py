# -*- coding: utf-8 -*-
"""
Created on Tue Nov 3 2020

@author: Sylvain Friot
Based on https://pytorch.org/docs/stable/_modules/torchvision/models/inception.html#inception_v3
aux_logits = False car pas utilisé dans notre cas. Retiré du code.
transform_input = False car on la transformation des input est déjà faite lors de leur chargement.
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    import sf_features_extraction as sffe
else:
    import modules_perso.sf_features_extraction as sffe


__all__ = ['Inception3', 'inception_v3']

# url de inception_v3 pré-entraîné
model_inceptionv3_url = \
    "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth"


class Inception3(sffe.FeatureExtraction):

    def __init__(self, name, num_classes=1000, inception_blocks=None,
                 max_num_features=None, folder_saves=None, init_weights=True):
        self.model_name = name
        self.model_family = "inception"
        super().__init__(max_num_features, folder_saves)
        self.num_classes = num_classes

        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]

        self.features = nn.Sequential()
        self.features.add_module("Conv2d_1a_3x3", conv_block(3, 32, kernel_size=3, stride=2))
        self.features.add_module("Conv2d_2a_3x3", conv_block(32, 32, kernel_size=3))
        self.features.add_module("Conv2d_2b_3x3", conv_block(32, 64, kernel_size=3, padding=1))
        self.features.add_module("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2))
        self.features.add_module("Conv2d_3b_1x1", conv_block(64, 80, kernel_size=1))
        self.features.add_module("Conv2d_4a_3x3", conv_block(80, 192, kernel_size=3))
        self.features.add_module("maxpool2", nn.MaxPool2d(kernel_size=3, stride=2))
        self.features.add_module("Mixed_5b", inception_a(192, pool_features=32))
        self.features.add_module("Mixed_5c", inception_a(256, pool_features=64))
        self.features.add_module("Mixed_5d", inception_a(288, pool_features=64))
        self.features.add_module("Mixed_6a", inception_b(288))
        self.features.add_module("Mixed_6b", inception_c(768, channels_7x7=128))
        self.features.add_module("Mixed_6c", inception_c(768, channels_7x7=160))
        self.features.add_module("Mixed_6d", inception_c(768, channels_7x7=160))
        self.features.add_module("Mixed_6e", inception_c(768, channels_7x7=192))
        self.features.add_module("Mixed_7a", inception_d(768))
        self.features.add_module("Mixed_7b", inception_e(1280))
        self.features.add_module("Mixed_7c", inception_e(2048))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, self.num_classes)
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
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, include_feat=False):
        feat = []
        x = self.features[0](x)
        previous_x = x

        for layer in self.features[1:]:
            x = layer(x)
            if x.size(2) < previous_x.size(2):  # reduction level, previous_x is a layer before reduction
                feat.append(previous_x)
            previous_x = x
        feat.append(x)  # append last layer of features extraction

        out = self.avgpool(x)
        out = self.dropout(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        if include_feat:
            return out, feat
        return out


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)
        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3),
                     self.branch3x3_2b(branch3x3),
                     ]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl),
                        self.branch3x3dbl_3b(branch3x3dbl),
                        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x num_classes
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def inception_v3(num_classes=1000, pretrained=False,
                 max_num_features=None, folder_saves=None,**kwargs):
    model = Inception3("inception_v3", num_classes=num_classes,
                       max_num_features=max_num_features, folder_saves=folder_saves,
                       **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_inceptionv3_url)
        if num_classes != 1000:
            state_dict.pop("fc.weight")
            state_dict.pop("fc.bias")
        temp_dict = {}
        for k, v in state_dict.items():
        	new_key = k.replace("Conv", "features.Conv")
        	new_key = new_key.replace("Mixed", "features.Mixed")
        	if not k.startswith("AuxLogits"):
        		temp_dict[new_key] = v
        model.load_state_dict(temp_dict, strict=False)
    return model


"""
Evolution des tailles en sortie pour mémoire
x : N x 3 x 299 x 299
Conv2d_1a_3x3 :  N x 32 x 149 x 149
Conv2d_2a_3x3 : N x 32 x 147 x 147
Conv2d_2b_3x3 : N x 64 x 147 x 147
maxpool1 : N x 64 x 73 x 73
Conv2d_3b_1x1 : N x 80 x 73 x 73
Conv2d_4a_3x3 : N x 192 x 71 x 71
maxpool2 : N x 192 x 35 x 35
Mixed_5b : N x 256 x 35 x 35
Mixed_5c : N x 288 x 35 x 35
Mixed_5d : N x 288 x 35 x 35
Mixed_6a : N x 768 x 17 x 17
Mixed_6b : N x 768 x 17 x 17
Mixed_6c : N x 768 x 17 x 17
Mixed_6d : N x 768 x 17 x 17
Mixed_6e : N x 768 x 17 x 17
aux_defined = self.training and self.aux_logits
if aux_defined: aux = self.AuxLogits else: aux = None -> N x 768 x 17 x 17
Mixed_7a : N x 1280 x 8 x 8
Mixed_7b : N x 2048 x 8 x 8
Mixed_7c : N x 2048 x 8 x 8
avgpool : N x 2048 x 1 x 1
dropout : N x 2048 x 1 x 1
flatten : N x 2048
fc : N x num_classes
"""
