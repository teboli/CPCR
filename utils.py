import os

import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import fftconvolve
from torchvision import transforms

from skimage import transform

import matplotlib.pyplot as plt


def to_tensor(array, device):
    tensor = torch.from_numpy(array)
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    else:
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
    return tensor.to(device).float()


def to_numpy(tensor):
    tensor = tensor.squeeze()
    if len(tensor.shape) == 3:
        tensor = tensor.permute(1, 2, 0)
    array = tensor.detach().cpu().numpy()
    return array


def random_crop(img, ps):
    h, w = img.shape[-2:]
    i0 = np.random.randint(h - ps - 1)
    j0 = np.random.randint(w - ps - 1)
    return img[...,i0:i0+ps, j0:j0+ps]


def crop_valid(img, ker):
    hks = ker.shape[-1] // 2
    return F.pad(img, (-hks, -hks, -hks, -hks))


def save_blur_and_sharp(savepath, x, hat_x, y):
    plt.imsave(savepath + 'x.jpg', to_numpy(x), cmap='gray')
    plt.imsave(savepath + 'y.jpg', to_numpy(y.clamp(0, 1)), cmap='gray')
    plt.imsave(savepath + 'hat_x.jpg', to_numpy(hat_x.clamp(0, 1)), cmap='gray')


def create_mask(tensor, min, max):
    mask1 = tensor > min
    mask2 = tensor <= max
    return mask1 * mask2


def get_labels(kmag, kori, mag_max=25):
    # for circular coherence of labels
    kori = torch.fmod(kori, 180)
    mask = create_mask(kori, 174, 177)
    kori[mask] = 174
    kori[kori > 177] = 0

    # for magnitude
    kmag[kmag > mag_max] = mag_max

    # find the labels
    inte_ori = mag_max // 2
    inte_mag = 2
    idxq = torch.round(kori / 6)
    tmp = kmag - 1
    tmp[tmp < 0] = 0
    idxm = torch.round(tmp / inte_mag)

    kori_q = idxq * inte_ori
    kmag_q = 1 + idxm * inte_mag

    labels = idxq * inte_ori + idxm + 1
    labels[kmag_q==1] = 1
    return labels-1


def get_square_kernel(kernel):
    hk, wk = kernel.shape
    d = max(hk-wk,wk-hk)
    i = np.argmin([hk-wk,wk-hk])
    if i == 0:
        kernel = np.pad(kernel, [[d//2,d//2], [0,0]], mode='edge')
    else:
        kernel = np.pad(kernel, [[0,0], [d//2,d//2]], mode='edge')
    return kernel

# from kruse et al. 17
def pad_for_kernel(img, kernel, mode):
    p = [(d-1)//2 for d in kernel.shape]
    padding = [p, p] + (img.ndim-2)*[(0, 0)]
    return np.pad(img, padding, mode)


# from kruse et al. 17
def crop_for_kernel(img, kernel):
    p = [(d-1)//2 for d in kernel.shape]
    r = [slice(p[0], -p[0]), slice(p[1], -p[1])] + (img.ndim-2)*[slice(None)]
    return img[r]

# from kruse et al. 17
def pad_for_kernel_v2(img, kernel, mode):
    p = [(int(1.8*d)-1)//2 for d in kernel.shape]
    padding = [p, p] + (img.ndim-2)*[(0, 0)]
    return np.pad(img, padding, mode)


# from kruse et al. 17
def crop_for_kernel_v2(img, kernel):
    p = [(int(1.8*d)-1)//2 for d in kernel.shape]
    r = [slice(p[0], -p[0]), slice(p[1], -p[1])] + (img.ndim-2)*[slice(None)]
    return img[r]


# from kruse et al. 17
def edgetaper_alpha(kernel, img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel, 1-i), img_shape[i]-1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z, z[0:1]], 0)
        v.append(1 - z/np.max(z))
    return np.outer(*v)


# from kruse et al. 17
def edgetaper(img, kernel, n_tapers=3):
    alpha = edgetaper_alpha(kernel, img.shape[0:2])
    _kernel = kernel
    if 3 == img.ndim:
        kernel = kernel[..., np.newaxis]
        alpha  = alpha[..., np.newaxis]
    for i in range(n_tapers):
        blurred = fftconvolve(pad_for_kernel(img, _kernel,'wrap'), kernel, mode='valid')
        img = alpha*img + (1-alpha)*blurred
    return img


def apply_prox(model):
    for i in range(model.n_iter):
        # normalize kernels
        if hasattr(model, 'weight'):
            mean = model.weight[i].data.view(model.N_c, -1).mean(1).view(-1,1,1,1)
            model.weight[i].data.sub_(mean)
            norm = model.weight[i].data.view(model.N_c, -1).norm(dim=1).view(-1,1,1,1)
            model.weight[i].data.div_(norm)
        # positivity constraints over lambd
        if hasattr(model, 'lambd'):
            model.lambd[i].data.clamp_(min=0)
        # positivity constraints over eta
        if hasattr(model, 'eta'):
            model.eta[i].data.clamp_(min=0)


def create_linear_kernels(n_filters, mag_max, savepath):
    kernels = np.zeros(n_filters,mag_max,mag_max)
    kernels[0,mag_max//2,mag_max//2] = 1
    i = 1
    for o in range(0,180,6):
        for m in range(0,mag_max+1,2):
            ker = np.zeros(mag_max, mag_max)
            ker[mag_max//2, max_max//2 - (m//2):mag_max//2 + (m//2) + 1] = 1
            ker = transform.rotate(ker, o)
            kernels[i] = ker / ker.sum()
            i += 1
    kernels = torch.from_array(kernels).float()
    torch.save(kernels, savepath)


def get_channelwise_kernel(k):
    channel = k.shape[0]
    weight = torch.zeros(channel,channel,*k.shape[-2:], device=k.device)
    weight[range(channel), range(channel)] = k
    return weight


def polar2cart(motion):
    out = torch.zeros_like(motion)
    out[:,0] = motion[:,0] * torch.cos(motion[:,1].mul(np.pi/180))
    out[:,1] = motion[:,0] * torch.sin(motion[:,1].mul(np.pi/180))
    return out


def cart2polar(motion):
    out = torch.zeros_like(motion)
    u = motion[:,0].float()
    v = motion[:,1].float()
    out[:,0] = (u.pow(2) + v.pow(2)).sqrt()
    # u[u == 0] = 1e-16
    # o = torch.atan(v.div(u)).mul(180 / np.pi)
    o = torch.atan2(v,u).mul(180/np.pi)
    o[o<0] = o[o<0] + 180
    # o = o + 360
    # out[:,1] = o.fmod(180)
    out[:,1] = o
    return out


def motion2labels(motion):
    u = motion[:,0]
    v = motion[:,1]

    sign_v = v.sign()
    sign_v[sign_v == 0] = 1
    u *= sign_v
    v *= sign_v

    # labels_v = v.abs().round()
    labels_v = v.round()
    labels_v[labels_v < 0] = 0
    labels_v[labels_v > 38] = 38
    labels_u = u.round()
    labels_u[labels_u < -36] = -36
    labels_u[labels_u > 37] = 37
    labels_u.add_(36)

    return labels_u, labels_v


def labels2motion(labels_u, labels_v):
    motion = torch.zeros_like(labels_u).unsqueeze(1).expand(-1,2,-1,-1).float()
    # print('labels_u', labels_u[0,10,10].item(), 'labels_v', labels_v[0,10,10].item())
    # pdb.set_trace()
    motion[:,0] = labels_u.float() - 36
    motion[:,1] = labels_v.float()

    return motion


def conj(A):
    B = A.clone()
    B[...,-1] *= -1
    return B


def square_modulus(A):
    return A[...,0].pow(2).unsqueeze(-1) + A[...,1].pow(2).unsqueeze(-1)
#     return A.norm(2, -1).unsqueeze(-1)


def prod(A, B):
    AB1 = A[...,0]*B[...,0] - A[...,1]*B[...,1]
    AB2 = A[...,0]*B[...,1] + A[...,1]*B[...,0]
    return torch.cat([AB1.unsqueeze(-1), AB2.unsqueeze(-1)], dim=-1)


def div(A, B):
    mod2_B = square_modulus(B)
    prod_AB = prod(A, conj(B))
    return prod_AB / mod2_B


def psf2otf(psf, img_shape):
    # inspired from 
    # https://github.com/aboucaud/pypher/blob/master/pypher/pypher.py
    # psf: (h_psf, w_psf)
    # img_shape: (h_otf, w_otf)

    # build padded array
    psf_shape = psf.shape
    h_pad = (img_shape[0] - psf_shape[0]) // 2
    w_pad = (img_shape[1] - psf_shape[1]) // 2
    if h_pad > 0:
        psf = F.pad(psf.unsqueeze(0).unsqueeze(0), (w_pad, w_pad, h_pad, h_pad))
    if psf.shape[-1] < img_shape[-1]:
        psf = F.pad(psf, (0, 1, 0, 0))
    if psf.shape[-2] < img_shape[-2]:
        psf = F.pad(psf, (0, 0, 0, 1))
    psf = psf.squeeze(0)
    
    # circular shift
    for axis, axis_size in enumerate(img_shape[1:]):
        psf = torch.roll(psf, -int(axis_size) // 2 + 1, axis)
        
    # compute OTF
    otf = torch.rfft(psf, 2, onesided=False)
    
    return otf


# def otf2psf(otf, psf_shape):
#     # otf: (h_otf, w_otf)
#     # psf_shape: (h_psf, w_psf)
    
#     # compute PSF
#     psf = np.fft.ifft2(otf)
    
#     # circular shift
#     otf_shape = otf.shape
#     for axis in [1, 0]:
#         axis_size = otf_shape[axis]
#         psf = np.roll(psf, int(axis_size) // 2, axis=axis)
    
#     # build cropped kernel
#     h_pad = (otf_shape[0] - psf_shape[0]) // 2
#     w_pad = (otf_shape[1] - psf_shape[1]) // 2
#     if h_pad > 0:
#         psf = psf[h_pad:-h_pad, w_pad:-w_pad]
    
#     return psf


def fftconvolve(x, k, correlation=False):
    prev_sizex = x.shape
    x = x.squeeze().unsqueeze(0)
    prev_sizek = k.shape
    k = k.squeeze()
    sizex = x.shape[1:]
    otfk = psf2otf(k, sizex)
    if correlation:
        otfk = conj(otfk)
    xfft = torch.rfft(x, 2, onesided=False)
    yfft = prod(xfft, otfk)

    y = torch.irfft(yfft, 2, signal_sizes=sizex, onesided=False)
    y = y.view(prev_sizex).clone()
    return y 
