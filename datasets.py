import os
import random

import torch
import torch.nn.functional as F
from torch.utils import data

import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from scipy.io import loadmat

from utils import random_crop


class UniformTrainDataset(data.Dataset):
    def __init__(self, root, datasize, ps, train=False, transform=False):
        folder = 'tr' if train else 'va'
        datapath = os.path.join(root, 'data_' + folder + '_%05d_ps_%03d.pt' % (datasize, ps))
        self.data = torch.load(datapath)
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        if self.train:
            x = self.data['x'][index]

            if self.transform:
                # lr flip
                if np.random.rand(1) < 0.5:
                    x = x.flip(-1)
                # ud flip
                if np.random.rand(1) < 0.5:
                    x = x.flip(-2)
                # rotation
                if np.random.rand(1) < 0.5:
                    x = torch.rot90(x, 1, [-2, -1])
                # random crop
                ps = min(x.shape[-1], x.shape[-2]) - 10
                x = random_crop(x, ps)
            idx_ke = np.random.randint(len(self.data['k']))
            k = self.data['k'][idx_ke]
            d = self.data['d'][idx_ke]
            hks = k.shape[-1] // 2
            padding = (hks, hks, hks, hks)
            y = F.conv2d(F.pad(x, padding, 'reflect'), k.unsqueeze(0)).squeeze(0)
            x = x.squeeze(0)
            return x, y, k, d
        else:
            x = self.data['x'][index]
            k = self.data['k'][index]
            d = self.data['d'][index]
            hks = k.shape[-1] // 2
            padding = (hks, hks, hks, hks)
            y = F.conv2d(F.pad(x, padding, 'reflect'), k.unsqueeze(0)).squeeze(0)
            x = x.squeeze(0)
            return x, y, k, d

    def __len__(self):
        return len(self.data['x'])


class NonUniformTrainDataset(data.Dataset):
    def __init__(self, root, datasize, ps, train=False, transform=False):
        folder = 'tr' if train else 'va'
        datapath = os.path.join(root, 'data_' + folder + '_%05d_ps_%03d.pt' % (datasize, ps))
        self.data = torch.load(datapath)
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        x = self.data['img'][index].unsqueeze(0)
        y = self.data['imblur'][index].unsqueeze(0)
        mag = self.data['mag'][index]
        ori = self.data['ori'][index]
        return x, y, mag, ori

    def __len__(self):
        return len(self.data['img'])


class LevinDataset(data.Dataset):
    def __init__(self, datapath):
        self.datapath = sorted(os.listdir(datapath))
        self.datapath = list(map(lambda x: os.path.join(datapath, x), self.datapath))

    def __getitem__(self, index):
        data = loadmat(self.datapath[index])
        x = torch.from_numpy(data['x']).unsqueeze(0).float()
        y = torch.from_numpy(data['y']).unsqueeze(0).float()
        f = torch.from_numpy(data['f']).unsqueeze(0).float()
        return x, y, f

    def __len__(self):
        return len(self.datapath)
