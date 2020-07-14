import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity, peak_noise_signal_ratio

import numpy as np

import utils


class MSELossGreedy(nn.Module):
    def __init__(self):
        super(MSELossGreedy, self).__init__()

    def forward(self, inputs, target):
        for i, input in enumerate(inputs):
            if i == 0:
                error = F.mse_loss(input, target)
            else:
                error = error + F.mse_loss(input, target)
        return error


class L1LossGreedy(nn.Module):
    def __init__(self):
        super(L1LossGreedy, self).__init__()

    def forward(self, inputs, target):
        for i, input in enumerate(inputs):
            if i == 0:
                error = F.l1_loss(input, target)
            else:
                error = error + F.l1_loss(input, target)
        return error


def psnr(input, target):
    err = []
    for b in range(input.shape[0]):
        im1 = input[b].permute(1,2,0).detach().cpu().numpy()
        im2 = target[b].permute(1,2,0).detach().cpu().numpy()
        err.append(peak_signal_noise_ratio(im1, im2))
    return torch.tensor(err, device=input.device).mean().item()


def ssim(input, target):
    err = []
    for b in range(input.shape[0]):
        im1 = input[b].permute(1,2,0).detach().cpu().numpy()
        im2 = target[b].permute(1,2,0).detach().cpu().numpy()
        err.append(structural_similarity(im1, im2, multichannel=True))
    return torch.tensor(err, device=input.device).mean().item()
