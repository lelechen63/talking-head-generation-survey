import os
import math
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch_layer_normalization import LayerNormalization

import pdb
import utils


class MidTemporalConv(nn.Module):
    def __init__(self, opt):
        super(MidTemporalConv, self).__init__()
        self.opt = opt
        self.inputDim = 512
        self.backend_conv1 = nn.Sequential(
            nn.Conv1d(self.inputDim, 2*self.inputDim, 5, 2, 0, bias=False),
            nn.BatchNorm1d(2*self.inputDim),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(2*self.inputDim, 4*self.inputDim, 5, 2, 0, bias=False),
            nn.BatchNorm1d(4*self.inputDim),
            nn.ReLU(True),
        )
        self.backend_conv2 = nn.Sequential(
            nn.Linear(4*self.inputDim, self.inputDim),
            nn.BatchNorm1d(self.inputDim),
            nn.ReLU(True),
            nn.Linear(self.inputDim, opt.out_channel)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feat):
        feat = feat.transpose(1, 2)
        feat = self.backend_conv1(feat)
        feat = torch.mean(feat, 2)
        feat = self.backend_conv2(feat)

        return feat


class Pass(nn.Module):
    def __init__(self, opt):
        super(Pass, self).__init__()
        self.opt = opt

    def forward(self, feat):

        return feat
