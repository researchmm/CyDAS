# --------------------------------------------------------
# Copyright (c) 2019 Jianyuan Guo (guojianyuan1@huawei.com)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, constant_init, xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule

from .auto_neck.build_neck import build_search_neck


@NECKS.register_module
class SearchPAFPN(nn.Module):
    r""" PAFPN Arch
        TBS      TD      TBS      BU
    C5 -----> C5     P5 -----> N5    N5
            
    C4 -----> C4     P4 -----> N4    N4
            
    C3 -----> C3     P3 -----> N3    N3
           
    C2 -----> C2     P2 -----> N2    N2
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 pa_kernel=3,
                 search_neck=None):
        super(SearchPAFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False
        self.pa_kernel = pa_kernel

        self.SearchNeck = build_search_neck(search_neck)

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.pa_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level - 1):  # Faster (0,3) one-stage (1,3)
            if pa_kernel > 0:    
                pa_conv = ConvModule(
                    out_channels, out_channels, pa_kernel,
                    padding=(pa_kernel-1)//2, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                    activation=activation, inplace=True)
                
                self.pa_convs.append(pa_conv)

        # add extra conv layers (e.g., RetinaNet); one-stage 5-4+1
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            self.fpn_convs = nn.ModuleList()
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channel = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channel = out_channels
                extra_fpn_conv = ConvModule(
                    in_channel, out_channels, 3,
                    stride=2, padding=1, conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg, activation=self.activation, inplace=True)                
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        # inputs [C2, C3, C4, C5]
        assert len(inputs) == len(self.in_channels)

        # build top-down laterals
        laterals = self.SearchNeck(inputs[self.start_level:], 1)

        used_backbone_levels = len(laterals)  # Faster rcnn:4; one-stage:3

        # Top-down path
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
        
        laterals_mid = self.SearchNeck(laterals, 2)
        
        # Bottom-up path
        # build outputs
        if self.pa_kernel > 0:
            outs = [laterals_mid[0]]
            for i in range(0, self.backbone_end_level - self.start_level - 1):  # Faster: [0,3]
                tmp = F.max_pool2d(outs[i], 2, stride=2) + laterals_mid[i + 1]
                outs.append(self.pa_convs[i](tmp))
        else:
            outs = laterals_mid

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[0](orig))
                else:
                    outs.append(self.fpn_convs[0](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i-used_backbone_levels](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i-used_backbone_levels](outs[-1]))
        return tuple(outs), None
