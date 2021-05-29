import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, constant_init, kaiming_init, normal_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import ResLayer
        

class BasicTransBlock(nn.Module):
    # expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 convlayers_num=2,
                 conv_cfg=None,
                 norm_cfg=None):
        super(BasicTransBlock, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.convlayers_num = convlayers_num

        self.trans_convs = nn.ModuleList()
        for i in range(self.convlayers_num):
            chn = in_channels if i == 0 else out_channels
            self.trans_convs.append(
                ConvModule(
                chn,
                out_channels,
                3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg))

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.trans_convs:
            normal_init(m.conv, std=0.01)

    def forward(self, x, y=None):
        """Forward function."""

        out = x
        for trans_conv in self.trans_convs:
            out = trans_conv(out)

        return out

class BasicCrossBlock(nn.Module):
    # expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None):
        super(BasicCrossBlock, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        chn1 = in_channels*2
        chn2 = in_channels*2
        self.conv1 = ConvModule(
            chn1,
            out_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

        self.conv2 = ConvModule(
            chn2,
            out_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.conv1.conv, std=0.01)
        normal_init(self.conv2.conv, std=0.01)

    def forward(self, x, y):
        """Forward function."""

        xy = torch.cat( (x, y), dim=-3)
        out1 = self.conv1(xy)
        out2 = self.conv2(xy)

        return out1, out2
