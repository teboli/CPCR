import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

import numpy as np

import datasets, networks, utils, loss, kernels

import os
import sys
import time
import tqdm


def train(loader, model, optimizer, criterion, epoch, d1, d2, blind, noise_level):
    running_l1 = 0
    train_l1 = 0
    model.train(True)

    k1 = model.weight[0].unsqueeze(0).expand(loader.batch_size, -1, -1, -1)
    k2 = model.weight[1].unsqueeze(0).expand(loader.batch_size, -1, -1, -1)
    d1 = d1.expand(loader.batch_size, -1, -1, -1)
    d2 = d2.expand(loader.batch_size, -1, -1, -1)

    for i, data in tqdm.tqdm(enumerate(loader)):
        x, y, k, d = data
        x = x.to(device)
        y = y.to(device)
        k = k.to(device)
        d = d.to(device)

        if blind:
            nl = (noise_level - 0.5) * np.random.rand(1) + 0.5
        else:
            nl = noise_level
        nl = float(nl) / 255
        y += nl*torch.randn_like(y)
        y = y.clamp(0, 1)
        y.requires_grad_()

        optimizer.zero_grad()

        hat_x = model(y, k ,d, k1, k2, d1, d2)

        error = criterion(hat_x, x)
        error.backward()

        optimizer.step()

        # compute running loss
        hat_x[-1] = utils.crop_valid(hat_x[-1], k)
        x = utils.crop_valid(x, k)
        y = utils.crop_valid(y, k)

        running_l1 += F.l1_loss(hat_x[-1], x).item()
        train_l1 += F.l1_loss(hat_x[-1], x).item()

        if (i+1) % 500 == 0:
            running_l1 /= 500
            print('    Running loss %2.5f' % (running_l1))
            running_l1 = 0

    return train_l1 / len(loader)



def validate(loader, model, epoch, d1, d2, blind, noise_level):
    val_psnr = 0
    val_ssim = 0
    val_l1 = 0
    model.train(False)

    k1 = model.weight[0].unsqueeze(0).expand(loader.batch_size, -1, -1, -1)
    k2 = model.weight[1].unsqueeze(0).expand(loader.batch_size, -1, -1, -1)
    d1 = d1.expand(loader.batch_size, -1, -1, -1)
    d2 = d2.expand(loader.batch_size, -1, -1, -1)

    # pre-create noise levels
    if blind:
        nls = np.linspace(0.5, noise_level, len(loader))
    else:
        nls = noise_level*np.ones(len(loader))

    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(loader)):
            x, y, k, d = data
            x = x.to(device)
            y = y.to(device)
            k = k.to(device)
            d = d.to(device)

            nl = nls[i] / 255
            y += nl*torch.randn_like(y)
            y = y.clamp(0, 1)

            hat_x = model(y, k ,d, k1, k2, d1, d2)

            hat_x = utils.crop_valid(hat_x, k)
            x = utils.crop_valid(x, k)
            y = utils.crop_valid(y, k)

            val_psnr += loss.psnr(hat_x, x)
            val_ssim += loss.ssim(hat_x, x)
            val_l1 += F.l1_loss(hat_x, x).item()

    return val_psnr / len(loader), val_ssim / len(loader), val_l1 / len(loader)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = options.options()
    opts = parser.parse_args()

    ### data loader
    # datapath = '/sequoia/data2/teboli/irc_nonblind/data/training_uniform'
    datapath = opts.datapath
    savepath = './results'
    savepath = os.path.join(savepath, 'uniform_T_%02d_S_%02d' % (opts.n_out, opts.n_in))
    if blind:
        savepath += '_blind_0.5_to_%2.2f' % (255 /100 * opts.noise_level)
    else:
        savepath += '_nonblind_%2.2f' % (255 / 100 * opts.noise_level)
    os.makedirs(savepath, exist_ok=True)
    modelpath = os.path.join(savepath, 'weights')
    os.makedirs(modelpath, exist_ok=True)
    dt_tr = datasets.UniformTrainDataset(opts.datapath, opts.datasize, opts.ps, train=True, transform=True)
    dt_va = datasets.UniformTrainDataset(opts.datapath, opts.datasize, ps, train=False)

    loader_tr = DataLoader(dt_tr, batch_size=opts.batch_size, shuffle=True, num_workers=4)
    loader_va = DataLoader(dt_va, batch_size=opts.batch_size, shuffle=False, num_workers=4)


    ######## STAGEWISE TRAINING ########
    ### model
    model = networks.LCHQS(opts.n_out, opts.n_in)
    if opts.load_epoch > 0:
        filename = 'epoch_%03d.pt' % (opts.load_epoch)
        state_dict_path = os.path.join(modelpath, filename)
        model.load_state_dict(torch.load(state_dict_path))

    ### inverse filters
    k1 = model.weight[0].data
    d1 = kernels.compute_inverse_filter_basic(k1, opts.lambd, 31).unsqueeze(0)
    k2 = model.weight[1].data
    d2 = kernels.compute_inverse_filter_basic(k2, opts.lambd, 31).unsqueeze(0)

    ### optimizer
    optimizer = Adam(model.parameters(), lr=opts.lr)
    model = model.to(device)
    criterion = loss.L1LossGreedy()
    criterion = criterion.to(device)
    d1 = d1.to(device)
    d2 = d2.to(device)

    ### loop
    scores = np.zeros((4, 2*opts.n_epochs))
    if opts.load_epoch < opts.n_epochs:
        for epoch in range(opts.load_epoch, opts.n_epochs, 1):
            print('### Epoch %03d ###' % (epoch+1))
            train_l1 = train(loader_tr, model, optimizer, criterion, epoch, d1, d2, opts.blind, 255/100*opts.sigma)
            print('   TR L1: %2.5f' % (train_l1))
            val_psnr, val_ssim, val_l1 = validate(loader_va, model, epoch, d1, d2, opts.blind, 255/100*opts.sigma)
            print('   VA L1: %2.5f / PSNR: %2.2f / SSIM: %2.3f' % (val_l1, val_psnr, val_ssim))

            # save net
            if (epoch+1) % 15 == 0:
                filename = 'epoch_%03d.pt' % (epoch+1)
                torch.save(model.state_dict(), os.path.join(modelpath, filename))

            # save results
            scores[0, epoch] = train_l1
            scores[1, epoch] = val_l1
            scores[2, epoch] = val_psnr
            scores[3, epoch] = val_ssim
            np.save(savepath, scores)


    ######## END2END TRAINING ########
    loadpath = os.path.join(modelpath, 'epoch_%03d.pt' % opts.n_epochs)

    ### model
    model = networks.LCHQS(opts.n_out, opts.n_in)
    state_dict = torch.load(loadpath, map_location='cpu')
    model.load_state_dict(state_dict)
    if opts.load_epoch > opts.n_epochs:
        filename = 'epoch_%03d.pt' % (opts.load_epoch)
        state_dict_path = os.path.join(modelpath, filename)
        model.load_state_dict(torch.load(state_dict_path))

    ### inverse filters
    k1 = model.weight[0].data
    d1 = kernels.compute_inverse_filter_basic(k1, opts.lambd, 31).unsqueeze(0)
    k2 = model.weight[1].data
    d2 = kernels.compute_inverse_filter_basic(k2, opts.lambd, 31).unsqueeze(0)

    ### optimizer
    lr = lr / 10
    optimizer = Adam(model.parameters(), lr=opts.lr)
    model = model.to(device)
    criterion = nn.L1Loss()
    criterion = criterion.to(device)

    ### loop
    for epoch in range(max(n_epochs, load_epoch), 2*opts.n_epochs):
        print('### Epoch %03d ###' % (epoch+1))
        train_l1 = train(loader_tr, model, optimizer, criterion, epoch, d1, d2, opts.blind, opts.sigma)
        print('   TR L1: %2.5f' % (train_l1))
        val_psnr, val_ssim, val_l1 = validate(loader_va, model, epoch, d1, d2, opts.blind, opts.sigma)
        print('   VA L1: %2.5f / PSNR: %2.2f / SSIM: %2.3f' % (val_l1, val_psnr, val_ssim))

        # save net
        if (epoch+1) % 15 == 0:
            filename = 'epoch_%03d.pt' % (epoch+1)
            torch.save(model.state_dict(), os.path.join(modelpath, filename))

        # save results
        scores[0, epoch] = train_l1
        scores[1, epoch] = val_l1
        scores[2, epoch] = val_psnr
        scores[3, epoch] = val_ssim
        np.save(savepath, scores)
