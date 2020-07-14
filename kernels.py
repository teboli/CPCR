import torch
import torch.nn.functional as F

import numpy as np
import os

import utils


def filt2matrix_largerv1(f, sim, tdim):
    nch = f.shape[0]  # handle many kernels
    fdim = f.shape[-1]
    imdim = sim
    outdim = tdim + fdim - 1

    e = torch.zeros(outdim, outdim, device=f.device)
    e[outdim//2, outdim//2] = 1

    matrix = torch.zeros(nch, outdim**2, imdim**2, device=f.device)
    ker = torch.zeros(nch, tdim+2*(fdim-1), tdim+2*(fdim-1), device=f.device)

    co = 0
    for i in range(outdim):
        for j in range(outdim):
            ker.zero_()
            ker[:, i:i+fdim, j:j+fdim] = f
            matrix[:, co] = ker[:, fdim-1:fdim+tdim-1, fdim-1:fdim+tdim-1].contiguous().view(nch, -1)
            co += 1
    return matrix, e


def filt2matrix_largerv2(f, sim, tdim):
    nch = f.shape[0]  # handle many kernels
    fdim = f.shape[-1]
    imdim = sim
    outdim = tdim + fdim - 1

    e = torch.zeros(outdim, outdim, device=f.device)
    e[outdim//2, outdim//2] = 1

    matrix = torch.zeros(nch, outdim**2, imdim**2, device=f.device)
    ker = torch.zeros(nch, tdim+2*(fdim-1), tdim+2*(fdim-1), device=f.device)

    matrix = filt2matrix_larger(f, matrix, ker, sim, tdim)
    return matrix, e


def psf2otf(psf, img_shape):
    # build padded array
    psf_shape = psf.shape
    h_pad = (img_shape[0] - psf_shape[0]) // 2
    w_pad = (img_shape[1] - psf_shape[1]) // 2
    psf = psf.unsqueeze(0).unsqueeze(0)
    if h_pad > 0:
        psf = F.pad(psf, (w_pad, w_pad, h_pad, h_pad))
    if psf.shape[-1] < img_shape[-1]:
        psf = F.pad(psf, (0, 1, 0, 0))
    if psf.shape[-2] < img_shape[-2]:
        psf = F.pad(psf, (0, 0, 0, 1))
    psf = psf.squeeze(0)
    
    # circular shift
    for axis, axis_size in enumerate(img_shape):
        psf = torch.roll(psf, -int(axis_size) // 2+1, axis+1)

    # compute OTF
    otf = torch.rfft(psf, 2, onesided=False)
    return otf


def otf2psf(otf, psf_shape):
    # compute PSF
    psf = torch.irfft(otf, 2, onesided=False)[0]
    
    # circular shift
    otf_shape = otf.shape[1:]
    for axis in [1, 0]:
        axis_size = otf_shape[axis]
        psf = torch.roll(psf, int(axis_size) // 2, axis)
    
    # build cropped kernel
    h_pad = (otf_shape[0] - psf_shape[0]) // 2
    w_pad = (otf_shape[1] - psf_shape[1]) // 2
    if h_pad > 0:
        psf = psf[h_pad:-h_pad, w_pad:-w_pad]
    return psf


def compute_inverse_filter_basic(ker, eps, ps):
    """
    ft (nks, ps, ps)
    """ 
    nks, hks, wks = ker.shape
    hps = nks // 2

    mt, e = filt2matrix_largerv1(ker.flip(-1,-2), ps, ps)
    mat_pls = torch.einsum('aik,ajk->aij', [mt, mt])
    idx = e.view(-1).nonzero().item()

    minv = [torch.inverse(mat_pls[b] + eps*torch.eye(mt.size(1), device=ker.device)).unsqueeze(0) for b in range(nks)]
    minv = torch.cat(minv)
    ft = minv[:, idx].unsqueeze(1).bmm(mt)
    return ft.view(nks, ps, ps)


def compute_inverse_filter_penalized(ker, eps, ps, betas):
    """
    fts (len(betas), 3, ps, ps)
    """
    nks, hks, wks = ker.shape

    if wks < 3 or hks < 3:
        hei = max(3, hks)
        wid = max(3, wks)
        ker2 = torch.zeros(nks, hei, wid, device=ker.device)
        ker2[:, hei//2-hks//2:hei//2+hks//2+1, wid//2-wks//2:wid//2+wks//2+1] = ker
        ker = ker2
        _, hks, wks = ker.shape
    centx = wks//2
    centy = hks//2
    hps = wks // 2

    grad_y = torch.zeros(1, 3, 3, device=ker.device)
    grad_y[0, 1, 0] = -1
    grad_y[0, 1, 1] = 1
    grad_x = grad_y.transpose(1, 2)

    grad = torch.zeros(2, hks, wks, device=ker.device)
    grad[0, centx-1:centx+2, centy-1:centy+2] = grad_y
    grad[1, centx-1:centx+2, centy-1:centy+2] = grad_x
    
    
    mt, e = filt2matrix_largerv1(ker.flip(-1, -2), ps, ps)
    mtt = torch.einsum('aik,ajk->aij', [mt, mt])
    kfilt, _ = filt2matrix_largerv1(grad.flip(-1, -2), ps, ps)
    mat_pls = torch.einsum('aik,ajk->ij', [kfilt, kfilt])
    idx = e.view(-1).nonzero().item()

    inv_filters = {'beta':[], 'ker':[], 'fts':[]}
    inv_filters['beta'] = betas
    inv_filters['ker']  = [ker, grad]

    for beta in betas:
        minv = [torch.inverse(mtt[b]+beta*mat_pls+eps*torch.eye(mt.size(1), device=ker.device)).unsqueeze(0) for b in range(nks)]
        minv = torch.cat(minv)
        ft = torch.zeros(nks, 3, ps, ps, device=mt.device)
        ft[:, 0] = minv[:, idx].unsqueeze(1).bmm(mt).view(nks,ps,ps)
        ft[:, 1] = minv[:, idx].unsqueeze(1).bmm(kfilt[0].unsqueeze(0).mul(beta)).view(nks, ps, ps)
        ft[:, 2] = minv[:, idx].unsqueeze(1).bmm(kfilt[1].unsqueeze(0).mul(beta)).view(nks, ps, ps)
        inv_filters['fts'].append(ft)
    inv_filters['fts'] = torch.cat(inv_filters['fts'])
    return inv_filters


def compute_inverse_filter_basic_fft(ker, eps, ps):
    """
    ft (nks, ps, ps)
    """ 
    nks = ker.shape[0]
    device = ker.device
    inv_ker = []
    for n in range(nks):
        K = psf2otf(ker[n], (ps, ps))
        D = utils.conj(K) / (utils.prod(utils.conj(K), K).sum(-1, keepdim=True) + eps)
        d = otf2psf(D, (ps, ps))
        inv_ker.append(d)
    inv_ker = torch.cat(inv_ker)
    return inv_ker


def compute_inverse_filter_fft_penalized(ker, eps, ps, betas):
    """
    fts (len(betas), 3, ps, ps)
    """
    nks, hks, wks = ker.shape

    if wks < 3 or hks < 3:
        hei = max(3, hks)
        wid = max(3, wks)
        ker2 = torch.zeros(nks, hei, wid, device=ker.device)
        ker2[:, hei//2-hks//2:hei//2+hks//2+1, wid//2-wks//2:wid//2+wks//2+1] = ker
        ker = ker2
        _, hks, wks = ker.shape
    centx = wks//2
    centy = hks//2
    hps = wks // 2

    grad_y = torch.zeros(1, 3, 3, device=ker.device)
    grad_y[0, 1, 0] = -1
    grad_y[0, 1, 1] = 1
    grad_x = grad_y.transpose(1, 2)
    
    grad = torch.zeros(2, hks, wks, device=ker.device)
    grad[0, centx-1:centx+2, centy-1:centy+2] = grad_y
    grad[1, centx-1:centx+2, centy-1:centy+2] = grad_x

    # compute denom
    otfks = []
    for n in range(nks):
        otfks.append(psf2otf(ker[n], (ps, ps)))
    otfks.append(psf2otf(grad_y[0], (ps, ps)))
    otfks.append(psf2otf(grad_x[0], (ps, ps)))
    
    mod2_otfks = []
    for n in range(nks+2):
        K = otfks[n]
        mod2_otfks.append(utils.prod(utils.conj(K), K))
    sum_mod2 = torch.stack(mod2_otfks).sum(0)
    
    inv_filters = {'beta':[], 'ker':[], 'fts':[]}
    inv_filters['beta'] = betas
    inv_filters['ker']  = torch.cat([ker, grad])

    for beta in betas:
        denom = sum_mod2.mul(beta) + eps
        ft = torch.zeros(nks, 3, ps, ps, device=ker.device)
        for n in range(nks):
            K = otfks[n]
            D = utils.conj(K) / denom
            d = otf2psf(D, (ps, ps))
            ft[n, 0] = d
            K = otfks[-2]
            D = utils.conj(K) / denom
            d = otf2psf(D, (ps, ps))
            ft[n, 1] = d
            K = otfks[-1]
            D = utils.conj(K) / denom
            d = otf2psf(D, (ps, ps))
            ft[n, 2] = d
            inv_filters['fts'].append(ft)
    inv_filters['fts'] = torch.cat(inv_filters['fts'])
    return inv_filters
