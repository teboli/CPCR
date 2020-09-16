import os
import numpy as np
from scipy import signal
from scipy.io import loadmat

import torch
from torch import nn
import torch.nn.functional as F

import conv

from layers.modules import *
from layers.functions import ConvMotionFunction


class Pooling(nn.Module):
    def __init__(self, n_in):
        super(Pooling, self).__init__()
        self.conv1 = nn.Conv2d(n_in, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 1, 3, padding=1)
        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.elu3 = nn.ELU()
        self.elu4 = nn.ELU()
        self.elu5 = nn.ELU()
        self.n_in = n_in

    def forward(self, inputs):
        x = torch.cat(inputs, 1)
        assert(x.size(1) == self.n_in)
        x = self.elu1(self.conv1(x))
        x = self.elu2(self.conv2(x))
        x = self.elu3(self.conv3(x))
        x = self.elu4(self.conv4(x))
        x = self.elu5(self.conv5(x))
        x = self.conv6(x)
        return x


class DenoiserGradient(nn.Module):
    def __init__(self, K=6):
        super(DenoiserGradient, self).__init__()
        self.K = K
        self.conv = [nn.Conv2d(1,64,5)]
        for k in range(K-2):
            self.conv.append(nn.Conv2d(64,64,3))
        self.conv.append(nn.Conv2d(64,1,3))
        self.conv = nn.ModuleList(self.conv)

    def forward(self, x):
        for k in range(self.K-1):
            hker = self.conv[k].weight.shape[-1]//2
            x = F.pad(x, (hker, hker, hker, hker), 'reflect')
            x = F.relu(self.conv[k](x))
        x = self.conv[-1](F.pad(x, (hker, hker, hker, hker), 'reflect'))
        return x


class HQS(nn.Module):
    def __init__(self, n_iter=5, n_in=2):
        super(HQS, self).__init__()
        self.n_iter = n_iter
        self.n_in = n_in
        self.beta = np.array([0,4**0,4,4**2,4**3,4**4,4**5,4**6,4**7,4**8])*1e-3 / 10 * 81
        self.beta = torch.from_numpy(self.beta).float()

    def forward(self, y, kmag, kori, labels):
        raise NotImplementedError()

    def init_grad(self):
        filters = torch.zeros(2,1,5,5)
        filters[1,0,2,2] = 1
        filters[1,0,1,2] = -1
        filters[0,0,2,2] = 1
        filters[0,0,2,1] = -1
        return nn.Parameter(filters, requires_grad=False)


class CHQS(HQS):
    def __init__(self, n_out=5, n_in=2, lambd=0.005):
        super(CHQS, self).__init__(n_out, n_in)
        self.weight = self.init_grad()
        self.lambd = lambd

    def forward(self, input, k, d):
        hks = k.shape[-1]//2
        hds = d.shape[-1]//2
        x_padding = (hks, hks, hks, hks)
        r_padding = (hds, hds, hds, hds)
        output = []
        for c in range(input.size(1)):
            y = input[:, c].unsqueeze(1)
            x = y.clone()
            for i in range(self.n_iter):
                # z update
                z = F.conv2d(F.pad(x, (2, 2, 2, 2), 'replicate'), self.weight)
                z = F.softshrink(z, self.lambd / max(1e-4, self.beta[i].item()))
                # x update
                for j in range(self.n_in):
                    r0 = y - F.conv2d(F.pad(x, x_padding, 'replicate'), k)
                    r1 = z - F.conv2d(F.pad(x, (2, 2, 2, 2), 'replicate'), self.weight)
                    r = torch.cat([r0, r1], dim=1)
                    r_pad = F.pad(r, r_padding, 'replicate')
                    for l in range(3):
                        x = x + F.conv2d(r_pad[:, l].unsqueeze(0), d[i, l].unsqueeze(0).unsqueeze(0))
                x = x.clamp(0, 1)
            output.append(x.clone())
        output = torch.cat(output, 1)
        return output

#### Main network for uniform deblurring ####
class LCHQS(HQS):
    def __init__(self, n_out=5, n_in=2, K=6):
        super(LCHQS, self).__init__(n_out, n_in)
        self.weight = self.init_grad()
        self.denoiser = nn.ModuleList([DenoiserGradient(K) for i in range(n_out)])
        self.pooling = nn.ModuleList([Pooling(3) for i in range(n_out*n_in)])

    def forward(self, input, k, d, k1, k2, d1, d2):
        hks = k.shape[-1]//2
        hds = d.shape[-1]//2
        hdds = d1.shape[-1]//2
        x_padding = (hks, hks, hks, hks)
        r_padding = (hds, hds, hds, hds)
        d_padding = (hdds, hdds, hdds, hdds)
        k_padding = (2, 2, 2, 2)
        output = []
        y = input
        x = y.clone()
        for i in range(self.n_iter):
            # z update
            x1 = conv.conv2d(x, k1, 'replicate')
            z1 = self.denoiser[i](x1)
            x2 = conv.conv2d(x, k2, 'replicate')
            z2 = self.denoiser[i](x2.transpose(-1, -2)).transpose(-1, -2)
            # x update
            for j in range(self.n_in):
                r0 = y - conv.conv2d(x, k, 'replicate')  # deblurred residual
                r0 = conv.conv2d(r0, d, 'replicate')
                r1 = z1 - conv.conv2d(x, k1, 'replicate')  # denoised residual
                r1 = conv.conv2d(r1, d1, 'replicate')
                r2 = z2 - conv.conv2d(x, k2, 'replicate')  # denoised residual
                r2 = conv.conv2d(r2, d2, 'replicate')
                x = x + self.pooling[i*self.n_in+j]([r0, r1, r2])
            output.append(x.clone())
        return output


class NUCHQS(HQS):
    def __init__(self, n_iter=5, n_in=2, N_c=2, lambd=0.005):
        super(NUCHQS, self).__init__(n_iter, n_in)

        self.weight = self.init_grad()
        kerpath1 = "./data/kers_grad.pt"
        kerpath2 = "./data/inverse_filter_nonuniform.pt"
        self.nu_conv1 = ConvMotion(self.weight)
        self.nu_conv2 = nn.ModuleList([ConvCls(kerpath2.format(i)) for j in range(n_in) for i in range(n_iter)])
        self.lambd = lambd

        N_l = self.nu_conv2[0].weight.shape[0]
        self.line = nn.ModuleList([Line(self.lambd, self.beta[i].item(), N_l, N_c) for i in range(n_iter)])

    def forward(self, y, kmag, kori, labels):
        x = y.clone()
        for i in range(self.n_iter):
            z = F.conv2d(F.pad(x, (2,2,2,2), 'reflect'), self.weight)
            z = self.line[i](z, labels)
            z = torch.cat([y, z], 1)
            for j in range(self.n_in):
                r = z - self.nu_conv1(x, kmag, kori)
                x = x + self.nu_conv2[i*self.n_in + j](r, labels)
            x = x.clamp(0, 1)
        return x


#### Main network for non-uniform deblurring ####
class NULCHQS(HQS):
    def __init__(self, weights, n_out=5, n_in=2, K=6):
        super(NULCHQS, self).__init__(n_out, n_in)
        self.weight = self.init_grad()
        self.denoiser = nn.ModuleList([DenoiserGradient(K) for i in range(n_out)])
        self.pooling = nn.ModuleList([Pooling(3) for i in range(n_out*n_in)])
        self.nu_conv = NUConv2d()
        self.inv_nu_conv = InvNUConv2d(weights)

    def forward(self, input, mag, ori, labels, k1, k2, d1, d2):
        hdds = d1.shape[-1]//2
        d_padding = (hdds, hdds, hdds, hdds)
        k_padding = (2, 2, 2, 2)
        output = []
        y = input
        x = y.clone()
        for i in range(self.n_iter):
            # z update
            x1 = F.conv2d(F.pad(x, k_padding, 'replicate'), k1)
            z1 = self.denoiser[i](x1)
            x2 = F.conv2d(F.pad(x, k_padding, 'replicate'), k2)
            z2 = self.denoiser[i](x2.transpose(-1, -2)).transpose(-1, -2)
            # x update
            for j in range(self.n_in):
                r0 = y - self.nu_conv(x, mag, ori)
                r0 = self.inv_nu_conv(r0, labels)
                r1 = z1 - F.conv2d(F.pad(x, k_padding, 'replicate'), k1)  # denoised residual
                r1 = F.conv2d(F.pad(r1, d_padding, 'replicate'), d1)
                r2 = z2 - F.conv2d(F.pad(x, k_padding, 'replicate'), k2)  # denoised residual
                r2 = F.conv2d(F.pad(r2, d_padding, 'replicate'), d2)
                x = x + self.pooling[i*self.n_in+j]([r0, r1, r2])
            output.append(x.clone())
        return output