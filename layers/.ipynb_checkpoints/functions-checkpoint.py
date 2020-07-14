import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

from irc import utils

import numpy as np
# import nu_conv


class ConvMotionFunction(Function):
    @staticmethod
    def forward(ctx, input, mag, ori):
        output = nu_conv.conv_motion_forward(input, mag, ori)
        ctx.save_for_backward(input, mag, ori)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        input, mag, ori = ctx.saved_tensors
        mcos = torch.zeros_like(input)
        msin = torch.zeros_like(input)
        grad_in = nu_conv.conv_motion_backward(grad_out, mag, ori, mcos, msin)
        return grad_in, None, None


class ConvClsFunction(Function):
    @staticmethod
    def forward(ctx, input, labels, labels_unique, weight, bias):
        output = nu_conv.conv_cls_forward(input, labels, weight, bias)
        ctx.save_for_backward(input, labels, labels_unique, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        input, labels, labels_unique, weight, bias = ctx.saved_tensors
        grad_in = nu_conv.conv_cls_backward1(grad_out, labels, weight)
        if ctx.needs_input_grad[3]:
            grad_f = nu_conv.conv_cls_backward2(grad_out, input, labels, weight, labels_unique)
        return grad_in, None, None, grad_f, None


class InvConvMotionFunction(Function):
    @staticmethod
    def forward(ctx, input, labels, labels_unique, weight, bias):
        output = nu_conv.inv_conv_motion_forward(input, labels, weight, bias)
        ctx.save_for_backward(input, labels, labels_unique, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        input, labels, labels_unique, weight, bias = ctx.saved_tensors
        grad_in = nu_conv.inv_conv_motion_backward(grad_out, labels, weight)
        return grad_in, None, None, None, None


class LineFunction(Function):
    @staticmethod
    def forward(ctx, input, labels, weightx, weighty):
        output = nu_conv.line_forward(weightx, weighty, input, labels)
        ctx.save_for_backward(input, output, labels, weightx, weighty)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        input, output, labels, weightx, weighty = ctx.saved_tensors
        grad_in = nu_conv.line_backward1(weightx, weighty, input, grad_out, labels)
        if ctx.needs_input_grad[3]:
            grad_y = nu_conv.line_backward2(weightx, weighty, input, output, labels)
        return grad_in, None, None, grad_y
