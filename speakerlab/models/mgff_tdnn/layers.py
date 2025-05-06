
""" This implementation is referred from 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)."""


import math

import torch
import torch.nn.functional as F
import torch.nn as nn


def length_to_mask(length, max_len=None, dtype=None, device=None):
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int):
    if stride > 1:
        n_steps = math.ceil(((L_in - kernel_size * dilation) / stride) + 1)
        L_out = stride * (n_steps - 1) + kernel_size * dilation
        padding = [kernel_size // 2, kernel_size // 2]

    else:
        L_out = (L_in - dilation * (kernel_size - 1) - 1) // stride + 1

        padding = [(L_in - L_out) // 2, (L_in - L_out) // 2]
    return padding


def get_nonlinear(nonlinear_str, channels):
    nonlinear = nn.Sequential()
    for name in nonlinear_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        else:
            raise ValueError(f'Not supported yet for {name} module')
    return nonlinear


class Conv1d(nn.Module):

    def __init__(
        self,
        out_channels,
        kernel_size,
        in_channels,
        stride=1,
        dilation=1,
        padding='same',
        groups=1,
        bias=True,
        padding_mode='reflect',
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        if self.padding == 'same':
            x = self._manage_padding(x, self.kernel_size, self.dilation,
                                     self.stride)

        elif self.padding == 'causal':
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == 'valid':
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got "
                + self.padding)

        wx = self.conv(x)

        return wx

    def _manage_padding(
        self,
        x,
        kernel_size: int,
        dilation: int,
        stride: int,
    ):
        L_in = x.shape[-1]
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)
        x = F.pad(x, padding, mode=self.padding_mode)

        return x


class TDNNBlock(nn.Module):
    """An implementation of TDNN.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        activation=nn.ReLU
    ):
        super(TDNNBlock, self).__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation
        )
        self.activation = activation()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class SEBlock(nn.Module):

    def __init__(self, in_channels, se_channels, out_channels):
        super(SEBlock, self).__init__()
        self.conv1 = Conv1d(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv1d(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths=None):
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L, device=x.device)
            mask = mask.unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)     # [1, 256, 1]

        s = self.relu(self.conv1(s))    # [1, 128, 1]
        s = self.sigmoid(self.conv2(s))     # [1, 512, 1]

        return s * x

def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)




class DenseLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = nn.BatchNorm1d(out_channels, affine=False)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class SoftmaxClassifier(nn.Module):
    def __init__(
        self,
        input_dim=512,
        out_neurons=1000,
    ):
        super(SoftmaxClassifier, self).__init__()

        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: [B, dim]
        # normalized
        x = F.linear(F.normalize(x), F.normalize(self.weight))
        return x
