import torch
from torch import nn
import torch.nn.functional as F
from .functions import ConvMotionFunction, InvConvMotionFunction, ConvClsFunction, LineFunction

import numpy as np


class NUConv2d(nn.Module):
    def __init__(self):
        super(NUConv2d, self).__init__()

    def forward(self, x, mag, ori):
        return ConvMotionFunction.apply(x, mag, ori)


class InvNUConv2d(nn.Module):
    def __init__(self, weight):
        super(InvNUConv2d, self).__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(self.weight.size(0)).float(), requires_grad=False)

    def forward(self, x, labels):
        labels_unique = labels.unique().float()
        return InvConvMotionFunction.apply(x, labels, labels_unique, self.weight, self.bias)


class ConvMotion(nn.Module):
    def __init__(self, weight):
        super(ConvMotion, self).__init__()
        self.weight = weight

    def forward(self, x, mag, ori):
        y0 = ConvMotionFunction.apply(x, mag, ori)
        y = F.conv2d(F.pad(x, (2,2,2,2), 'replicate'), self.weight)
        return torch.cat([y0,y], 1)


class ConvCls(nn.Module):
    def __init__(self, weightpath):
        super(ConvCls, self).__init__()
        self.weight = nn.Parameter(torch.load(weightpath))
        self.bias = nn.Parameter(torch.zeros(self.weight.size(0)), requires_grad=False)

    def forward(self, x, labels):
        labels_unique = labels.unique().float()
        return ConvClsFunction.apply(x, labels, labels_unique, self.weight, self.bias)


class Line(nn.Module):
    def __init__(self, lambd, beta, N_l, N_c):
        super(Line, self).__init__()
        self.weightx = nn.Parameter(torch.arange(-500, 510, 10, dtype=torch.float) / 255, requires_grad=False)
        l = lambd / max(beta, 1e-4)
        self.weighty = nn.Parameter(F.softshrink(self.weightx.view(1,1,-1).repeat(N_l, N_c, 1), l).contiguous())

    def forward(self, input, labels):
        return LineFunction.apply(input, labels, self.weightx, self.weighty)