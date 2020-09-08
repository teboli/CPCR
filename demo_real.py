import os
import glob

import torch
from torch.utils.data import DataLoader

import numpy as np
from skimage import img_as_float, metrics, io
from scipy.signal import correlate2d

import networks, datasets, utils, kernels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_out = 5
n_in = 2


# directly use model train on low noise, e.g., Levin dataset-like, for real images
modelpath = './data/lchqs_out_%02d_in_%02d_sigma_0.pt' % (n_out, n_in)
    
datapath = './data'
# image = 'hanzi'; pregu = 3e-2
# image = 'toy'; pregu = 2e-2
image = 'postcard'; pregu = 2e-2
    
# load model
model = networks.LCHQS(n_out, n_in)
state_dict = torch.load(modelpath, map_location='cpu')
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# load images and blur kernel
imblur = img_as_float(io.imread(os.path.join(datapath, image + '_blurry.png')))
ker = img_as_float(io.imread(os.path.join(datapath, image +  '_kernel.png')))
ker /= np.sum(ker)
ker = np.clip(ker, 0, 1)

# compute inverse filters for k1 and k2
k1 = model.weight[0]
k2 = model.weight[1]

d1 = kernels.compute_inverse_filter_basic_fft(k1, pregu, 31)
d2 = kernels.compute_inverse_filter_basic_fft(k2, pregu, 31)
d1 = d1.unsqueeze(0).unsqueeze(0).detach()
d2 = d2.unsqueeze(0).unsqueeze(0).detach()
d1 = d1.to(device)
d2 = d2.to(device)

k1 = k1.unsqueeze(0).to(device)
k2 = k2.unsqueeze(0).to(device)

# compute inverse filters for k0
y = utils.to_tensor(imblur, device)
k = utils.to_tensor(ker, device)
k = torch.rot90(k, 2, (-1,-2))
ps = int(2.0*k.shape[-1])
if ps % 2 == 0: ps +=1
d = kernels.compute_inverse_filter_basic_fft(k[0], pregu, ps)
d = d.unsqueeze(0).unsqueeze(0).to(device)

# inference
with torch.no_grad():
    hat_x = []
    for c in range(y.shape[1]):
        yc = y[:,c].unsqueeze(1)
        hat_x.append(model(yc, k, d, k1, k2, d1, d2)[-1])
    hat_x = torch.cat(hat_x, 1).clamp(0,1)
    
    pred = utils.to_numpy(hat_x)
    
io.imsave(os.path.join(datapath, image + '_result.png'), pred)