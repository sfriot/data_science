# -*- coding: utf-8 -*-
"""
Created on Tue Nov 4 2020

@author: Sylvain Friot
Based on https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
and https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import collections
import re
from functools import partial
from PIL import Image

import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    import sf_features_extraction as sffe
else:
    import modules_perso.sf_features_extraction as sffe
    

__all__ = ['EfficientNet', 'efficientnetb4', 'efficientnetb0',
           'efficientnetb7', 'efficientnetb2', 'efficientnetb3']


model_efficientnetb4_url = \
    "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth"
model_efficientnetb0_url = \
    "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"
model_efficientnetb7_url = \
    "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth"
model_efficientnetb2_url = \
    "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth"
model_efficientnetb3_url = \
    "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth"


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
    'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

# Set GlobalParams and BlockArgs's defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


class BlockDecoder(object):
    """Block Decoder for readability, from the official TensorFlow repository.
    """
    @staticmethod
    def _decode_block_string(block_string):
        """
        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.
        Returns:
            BlockArgs: The BloackArgs  namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=[int(options['s'][0])],
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio=float(options['se']) if 'se' in options else None,
            id_skip=('noskip' not in block_string))

    @staticmethod
    def decode(string_list):
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args


# 'SamePadding' functions make output size equal ceil(input size/stride).
# Only when stride equals 1, can the output size be the same as input size.
def get_same_padding_conv2d(image_size=None):
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw) # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, 
                        self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias,
                     self.stride, self.padding, self.dilation, self.groups)
        return x


def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.
    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.
    Returns:
        output_image_size: A list [H,W].
    """
    if input_image_size is None:
        return None
    if isinstance(input_image_size, int):
        image_height, image_width = input_image_size, input_image_size
    elif isinstance(input_image_size, list) or isinstance(input_image_size, tuple):
        image_height, image_width = input_image_size
    else:
        raise TypeError()
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def round_filters(filters, global_params):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth of global_params.
    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.
    Returns:
        new_filters: New filters number after calculating.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor # pay attention to this line when using min_depth
    # follow the formula transferred from official TensorFlow implementation
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters: # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.
    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.
    Returns:
        new repeat: New repeat number after calculating.
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """Drop connect.
    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.
    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, 'p must be in range of [0,1]'
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1],
                                dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.
    Args:
        block_args (namedtuple): BlockArgs
        global_params (namedtuple): GlobalParam
        image_size (tuple or list): [image_height, image_width].
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and \
            (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup,
                                       kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(in_channels=oup, out_channels=oup,
                                      groups=oup, kernel_size=k, stride=s,  # groups makes it depthwise
                                      bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = \
                max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels,
                                     kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup,
                                     kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup,
                                    kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters = self._block_args.input_filters
        output_filters = self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(sffe.FeatureExtraction):
    """EfficientNet model.
    Args:
        blocks_args : A list of BlockArgs to construct blocks.
        global_params : A set of GlobalParams shared between blocks.
    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)
    """
    def __init__(self, name, blocks_args, global_params,
                 max_num_features=None, folder_saves=None):
        self.model_name = name
        self.model_family = "efficientnet"
        super().__init__(max_num_features, folder_saves)
        assert isinstance(blocks_args, list), "blocks_args should be a list"
        assert len(blocks_args) > 0, "block args must be greater than 0"
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.num_classes = self._global_params.num_classes

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = self._global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels,
                                 kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params))
            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params,
                                            image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1: # modify block_args to keep same output size
                block_args = block_args._replace(
                        input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params,
                                                image_size=image_size))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels,
                                 kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self.num_classes)
        self._swish = MemoryEfficientSwish()

        self.get_num_features()

    def forward(self, inputs, include_feat=False):
        feat = []
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        previous_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if x.size(2) < previous_x.size(2):  # reduction level, previous_x is a layer before reduction
                feat.append(previous_x)
            previous_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        feat.append(x)

        # Pooling and final linear layer
        out = self._avg_pooling(x)
        if self._global_params.include_top:
            out = out.flatten(start_dim=1)
            out = self._dropout(out)
            out = self._fc(out)
        if include_feat:
            return out, feat
        return out

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels,
                                     kernel_size=3, stride=2, bias=False)


def get_efficientnet_params(width_coefficient=None, depth_coefficient=None,
                            image_size=None, dropout_rate=0.2,
                            drop_connect_rate=0.2, num_classes=1000,
                            include_top=True):
    # Blocks args for the whole model(efficientnet-b0 by default)
    # It will be modified in the construction of EfficientNet Class according to model
    blocks_args = ['r1_k3_s11_e1_i32_o16_se0.25',
                   'r2_k3_s22_e6_i16_o24_se0.25',
                   'r2_k5_s22_e6_i24_o40_se0.25',
                   'r3_k3_s22_e6_i40_o80_se0.25',
                   'r3_k5_s11_e6_i80_o112_se0.25',
                   'r4_k5_s22_e6_i112_o192_se0.25',
                   'r1_k3_s11_e6_i192_o320_se0.25']
    blocks_args = BlockDecoder.decode(blocks_args)
    global_params = GlobalParams(
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        image_size=image_size,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        drop_connect_rate=drop_connect_rate,
        depth_divisor=8,
        min_depth=None,
        include_top=include_top)
    return blocks_args, global_params


def efficientnetb4(num_classes=1000, pretrained=False, in_channels=3,
                   max_num_features=None, folder_saves=None,
                   url_pretrained=None, **kwargs):
    # in_channels : number of channels of input images ; 3=RGB
    w, d, s, p = (1.4, 1.8, 380, 0.4)
    blocks_args, global_params = get_efficientnet_params(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p,
            image_size=s, num_classes=num_classes)
    model = EfficientNet("efficientnetb4", blocks_args, global_params,
                         max_num_features=max_num_features, folder_saves=folder_saves,
                         **kwargs)
    model._change_in_channels(in_channels)
    if pretrained:
        if url_pretrained is None:
            state_dict = torch.load(model_efficientnetb4_url)
        else:
            state_dict = torch.load(url_pretrained)
        if num_classes != 1000:
            state_dict.pop("_fc.weight")
            state_dict.pop("_fc.bias")
        model.load_state_dict(state_dict, strict=False)
    return model


def efficientnetb0(num_classes=1000, pretrained=False, in_channels=3,
                   max_num_features=None, folder_saves=None,
                   url_pretrained=None, **kwargs):
    # in_channels : number of channels of input images ; 3=RGB
    w, d, s, p = (1.0, 1.0, 224, 0.2)
    blocks_args, global_params = get_efficientnet_params(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p,
            image_size=s, num_classes=num_classes)
    model = EfficientNet("efficientnetb0", blocks_args, global_params,
                         max_num_features=max_num_features, folder_saves=folder_saves,
                         **kwargs)
    model._change_in_channels(in_channels)
    if pretrained:
        if url_pretrained is None:
            state_dict = torch.load(model_efficientnetb0_url)
        else:
            state_dict = torch.load(url_pretrained)
        if num_classes != 1000:
            state_dict.pop("_fc.weight")
            state_dict.pop("_fc.bias")
        model.load_state_dict(state_dict, strict=False)
    return model


def efficientnetb7(num_classes=1000, pretrained=False, in_channels=3,
                   max_num_features=None, folder_saves=None,
                   url_pretrained=None, **kwargs):
    # in_channels : number of channels of input images ; 3=RGB
    w, d, s, p = (2.0, 3.1, 600, 0.5)
    blocks_args, global_params = get_efficientnet_params(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p,
            image_size=s, num_classes=num_classes)
    model = EfficientNet("efficientnetb7", blocks_args, global_params,
                         max_num_features=max_num_features, folder_saves=folder_saves,
                         **kwargs)
    model._change_in_channels(in_channels)
    if pretrained:
        if url_pretrained is None:
            state_dict = torch.load(model_efficientnetb7_url)
        else:
            state_dict = torch.load(url_pretrained)
        if num_classes != 1000:
            state_dict.pop("_fc.weight")
            state_dict.pop("_fc.bias")
        model.load_state_dict(state_dict, strict=False)
    return model


def efficientnetb2(num_classes=1000, pretrained=False, in_channels=3,
                   max_num_features=None, folder_saves=None,
                   url_pretrained=None, **kwargs):
    # in_channels : number of channels of input images ; 3=RGB
    w, d, s, p = (1.1, 1.2, 260, 0.3)
    blocks_args, global_params = get_efficientnet_params(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p,
            image_size=s, num_classes=num_classes)
    model = EfficientNet("efficientnetb2", blocks_args, global_params,
                         max_num_features=max_num_features, folder_saves=folder_saves,
                         **kwargs)
    model._change_in_channels(in_channels)
    if pretrained:
        if url_pretrained is None:
            state_dict = torch.load(model_efficientnetb2_url)
        else:
            state_dict = torch.load(url_pretrained)
        if num_classes != 1000:
            state_dict.pop("_fc.weight")
            state_dict.pop("_fc.bias")
        model.load_state_dict(state_dict, strict=False)
    return model


def efficientnetb3(num_classes=1000, pretrained=False, in_channels=3,
                   max_num_features=None, folder_saves=None,
                   url_pretrained=None, **kwargs):
    # in_channels : number of channels of input images ; 3=RGB
    w, d, s, p = (1.2, 1.4, 300, 0.3)
    blocks_args, global_params = get_efficientnet_params(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p,
            image_size=s, num_classes=num_classes)
    model = EfficientNet("efficientnetb3", blocks_args, global_params,
                         max_num_features=max_num_features, folder_saves=folder_saves,
                         **kwargs)
    model._change_in_channels(in_channels)
    if pretrained:
        if url_pretrained is None:
            state_dict = torch.load(model_efficientnetb3_url)
        else:
            state_dict = torch.load(url_pretrained)
        if num_classes != 1000:
            state_dict.pop("_fc.weight")
            state_dict.pop("_fc.bias")
        model.load_state_dict(state_dict, strict=False)
    return model


"""
FOR MEMORY

Values of w, d, s, p according to efficientnet model
(width_coefficient, depth_coefficient, resolution, dropout_rate)
'efficientnet-b0': (1.0, 1.0, 224, 0.2)
'efficientnet-b1': (1.0, 1.1, 240, 0.2)
'efficientnet-b2': (1.1, 1.2, 260, 0.3)
'efficientnet-b3': (1.2, 1.4, 300, 0.3)
'efficientnet-b4': (1.4, 1.8, 380, 0.4)
'efficientnet-b5': (1.6, 2.2, 456, 0.4)
'efficientnet-b6': (1.8, 2.6, 528, 0.5)
'efficientnet-b7': (2.0, 3.1, 600, 0.5)
'efficientnet-b8': (2.2, 3.6, 672, 0.5)
'efficientnet-l2': (4.3, 5.3, 800, 0.5)
  
URL of trained models
1. train with Standard methods
check more details in paper(EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks)
'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth'
'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth'
'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth'
'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth'
'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth'
'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth'
'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth'
'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth'

2. train with Adversarial Examples(AdvProp)
check more details in paper(Adversarial Examples Improve Image Recognition)
'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth'
'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth'
'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth'
'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth'
'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth'
'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth'
'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth'
'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth'
'efficientnet-b8': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth'
"""
