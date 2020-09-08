import os
import glob

import torch
from torch.utils.data import DataLoader

import numpy as np
from skimage import img_as_float, metrics, io
from scipy.io import loadmat

import networks, datasets, utils, kernels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_out = 5
n_in = 2
sigma = 1.
blind = False
pregu = 1e-2

if blind:
    modelpath = './data/nulchqs_out_%02d_in_%02d_blind.pt' % (n_out, n_in)
else:
    modelpath = './data/nulchqs_out_%02d_in_%02d_sigma_%d.pt' % (n_out, n_in, sigma)
    
datapath = './data'
    
# load model
d0 = torch.load(os.path.join(datapath, 'inverse_filters_nonunifom.pt'), map_location='cpu')
d0 = d0[:,0].unsqueeze(1)
model = networks.NULCHQS(d0, n_out, n_in)
state_dict = torch.load(modelpath, map_location='cpu')
model.load_state_dict(state_dict)
model.to(device)

# load images and blur kernel
img = img_as_float(io.imread(os.path.join(datapath, 'nonuniform_gt.png')))
imblur = img_as_float(io.imread(os.path.join(datapath, 'nonuniform_blurry.png')))
mfmap = loadmat(os.path.join(datapath, 'nonuniform_motion.mat'))['mfmap']
mag = torch.from_numpy(mfmap[...,0]).float().unsqueeze(0)
ori = torch.from_numpy(mfmap[...,1]).float().unsqueeze(0)
ori = (90-ori).add(360).fmod(180)  # simple adjustement of the orientations

k1 = model.weight[0]
k2 = model.weight[1]

d1 = kernels.compute_inverse_filter_basic(k1, pregu, 31)
d2 = kernels.compute_inverse_filter_basic(k2, pregu, 31)
d1 = d1.unsqueeze(0)
d2 = d2.unsqueeze(0)
d1 = d1.to(device)
d2 = d2.to(device)

k1 = k1.unsqueeze(0)
k2 = k2.unsqueeze(0)

y = utils.to_tensor(imblur, device)
y = y + sigma/255 * torch.randn_like(y)
y.clamp_(0,1)
labels = utils.get_labels(mag, ori)

# inference
with torch.no_grad():
    y = y.to(device)
    mag = mag.to(device)
    ori = ori.to(device)
    labels = labels.to(device)
    ori = ori / 180 * np.pi  # go to radians domain
    
    hat_x = []
    for c in range(y.shape[1]):
        yc = y[:,c].unsqueeze(1)
        hat_x = model(yc, mag, ori, labels, k1, k2, d1, d2)[-1]
    hat_x = torch.cat(hat_x, 1)
    
    pred = utils.to_numpy(hat_x)
    psnr = metrics.peak_signal_noise_ratio(pred, img)
    
print('PSNR is %2.2f' % psnr)
    
io.imsave(os.path.join(datapath, 'nonuniform_result.png'))