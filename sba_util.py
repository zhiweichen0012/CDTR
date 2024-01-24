import pdb

import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d
from mmcv.cnn import normal_init
import numpy as np


class OffsetExtractor(nn.Module):
    def __init__(self):
        super(OffsetExtractor, self).__init__()
    def forward(self, x):
        offset = self.conv_offset(x)
        return offset


class sba_module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(sba_module, self).__init__()
        self.conv_offset = nn.Conv2d(in_channels, 2 * 3 * 3, kernel_size=3, padding=1)
        init_offset = torch.Tensor(np.zeros([2 * 3 * 3, in_channels, 3, 3]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset)  

        self.deform_conv = DeformConv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, m_out_feat, m_ca):
        offset_kernel = self.conv_offset(m_out_feat * m_ca)
        deformable_output = self.deform_conv(x, offset_kernel)
        output = self.avgpool(deformable_output).squeeze(3).squeeze(2)
        return output

