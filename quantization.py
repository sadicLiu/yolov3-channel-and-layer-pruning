from __future__ import division

import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction
from torch.nn.parameter import Parameter

QuantizeScheme = namedtuple('QuantizScheme', ['method'])
DefaultScheme = QuantizeScheme('none')  # layer, bias

quant_hyp = {
    'weights': {'int': 4.0, 'range': 11.0, 'float': 11.0},
    'bias': {'int': 7.0, 'range': 8.0, 'float': 8.0},
    'activation': {'int': 7.0, 'range': 8.0, 'float': 8.0}
}


def compute_delta(input, bits, overflow_rate=0):
    # assert input.requires_grad==False, "tensor data must be no grad"
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    v = v.item()
    I_max = 2 ** (bits - 1) - 1
    si = math.floor(math.log(I_max / (v + 1e-12), 2))

    delta = math.pow(2.0, -si)
    return delta


class Quantization(InplaceFunction):
    @staticmethod
    def forward(ctx, input, inplace=False, scheme=DefaultScheme):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        with torch.no_grad():
            if scheme.method == 'bias':
                _int = 2 ** quant_hyp['bias']['int']
                _range = 2 ** quant_hyp['bias']['range']
                _float = 2 ** (-quant_hyp['bias']['float'])
                output = output.mul_(_range).round_().mul_(_float).clamp_(-_int, _int - _float)
                return output

            if scheme.method == 'layer':
                _int = 2 ** quant_hyp['weights']['int']
                _range = 2 ** quant_hyp['weights']['range']
                _float = 2 ** (-quant_hyp['weights']['float'])
                output = output.mul_(_range).round_().mul_(_float).clamp_(-_int, _int - _float)
                return output

            if scheme.method == 'bn':
                delta = compute_delta(input, 16, overflow_rate=0.001)
                float_ = - math.log(delta, 2)
                int_ = 7 - float_
                output = output / delta
                rounded = torch.floor(output) * delta
                clipped_value = torch.clamp(rounded, - 2 ** int_, 2 ** int_ - 2 ** (-float_))
                output = clipped_value
                return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None


def quantize(x, scheme=DefaultScheme, inplace=False):
    return Quantization().apply(x, inplace, scheme)


def weightquant(x, inplace=False):
    if x.dim() == 4:
        scheme = QuantizeScheme('layer')
    elif x.dim() == 2:
        scheme = QuantizeScheme('layer')
    elif x.dim() == 1:
        scheme = QuantizeScheme('bias')
    else:
        raise ValueError('dont support other shapes')

    return quantize(x, scheme=scheme, inplace=inplace)


class ActQuantize(InplaceFunction):
    @staticmethod
    def forward(ctx, input, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        with torch.no_grad():
            _int = 2 ** quant_hyp['activation']['int']
            _range = 2 ** quant_hyp['activation']['range']
            _float = 2 ** (-quant_hyp['activation']['float'])
            output = output.mul_(_range).floor_().mul_(_float).clamp_(-_int, _int - _float)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None, None, None


class ActQuant(nn.Module):
    def __init__(self, inplace=False):
        super(ActQuant, self).__init__()

    def forward(self, input):
        output = ActQuantize().apply(input, False)
        return output


class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.actquant = ActQuant()

    def forward(self, input):
        temp_weight = weightquant(self.weight)
        if self.bias is not None:
            temp_bias = weightquant(self.bias)
        else:
            temp_bias = None
        qinput = self.actquant(input)
        output = F.conv2d(qinput, temp_weight, temp_bias, self.stride,
                          self.padding, self.dilation, self.groups)

        qoutput = self.actquant(output)

        return qoutput


class QRelu(nn.ReLU):
    def __init__(self, inplace=False):
        super(QRelu, self).__init__(inplace=inplace)
        self.actquant = ActQuant()

    def forward(self, input):
        qinput = self.actquant(input)
        qoutput = F.leaky_relu(qinput, 0.125)
        qoutput = self.actquant(qoutput)
        return qoutput


class QLeakyReLu(nn.LeakyReLU):
    def __init__(self, negative_slope=0.125, inplace=False):
        super(QLeakyReLu, self).__init__(negative_slope=negative_slope, inplace=inplace)
        self.actquant = ActQuant()

    def forward(self, input):
        qinput = self.actquant(input)
        qoutput = F.leaky_relu(qinput, self.negative_slope)
        qoutput = self.actquant(qoutput)
        return qoutput


class IdentityBN(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(IdentityBN, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        return input


class QLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(QLinear, self).__init__(in_features, out_features, bias=bias)
        self.actquant = ActQuant()

    def forward(self, input):
        self.weight.data = weightquant(self.weight)
        if self.bias is not None:
            self.bias.data = weightquant(self.bias)
        output = F.linear(input, self.weight, self.bias)
        qoutput = self.actquant(output)

        return qoutput
