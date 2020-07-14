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
import argparse
import tqdm


def train_greedy(loader, model, optimizer, criterion, epoch, d1, d2):
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

        nl = (12.75 - 0.5) * np.random.rand(1) + 0.5
        nl = float(nl) / 255
        y += nl*torch.randn_like(y)
        y = y.clamp(0, 1)
        y.requires_grad_()

        optimizer.zero_grad()

        hat_x = model(y, k ,d, k1, k2, d1, d2)

        error = criterion(hat_x, x)
        error.backward()

        hat_x[-1] = utils.crop_valid(hat_x[-1], k)
        x = utils.crop_valid(x, k)
        y = utils.crop_valid(y, k)

        running_l1 += F.l1_loss(hat_x[-1], x).item()
        train_l1 += F.l1_loss(hat_x[-1], x).item()

        optimizer.step()

        if (i+1) % 500 == 0:
            running_l1 /= 500
            print('    Running loss %2.5f' % (running_l1))
            running_l1 = 0

    return train_l1 / len(loader)


def train_finetune(loader, model, optimizer, criterion, epoch, d1, d2):
    running_l1 = 0
    train_l1 = 0
    model.train(False)

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

        nl = (12.75 - 0.5) * np.random.rand(1) + 0.5
        nl = float(nl) / 255
        y += nl*torch.randn_like(y)
        y = y.clamp(0, 1)
        y.requires_grad_()

        optimizer.zero_grad()

        hat_x = model(y, k ,d, k1, k2, d1, d2)

        error = criterion(hat_x, x)
        error.backward()

        hat_x = utils.crop_valid(hat_x, k)
        x = utils.crop_valid(x, k)
        y = utils.crop_valid(y, k)

        running_l1 += F.l1_loss(hat_x, x).item()
        train_l1 += F.l1_loss(hat_x, x).item()

        optimizer.step()

        if (i+1) % 500 == 0:
            running_l1 /= 500
            print('    Running loss %2.5f' % (running_l1))
            running_l1 = 0

    return train_l1 / len(loader)


def validate(loader, model, epoch, d1, d2):
    val_psnr = 0
    val_ssim = 0
    val_l1 = 0
    model.train(False)

    k1 = model.weight[0].unsqueeze(0).expand(loader.batch_size, -1, -1, -1)
    k2 = model.weight[1].unsqueeze(0).expand(loader.batch_size, -1, -1, -1)
    d1 = d1.expand(loader.batch_size, -1, -1, -1)
    d2 = d2.expand(loader.batch_size, -1, -1, -1)

    # pre-create noise levels
    nls = np.linspace(0.5, 12.75, len(loader))

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_in', type=int, default=2)
    parser.add_argument('--n_out', type=int, default=5)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--datasize', type=int, default=3000)
    parser.add_argument('--ps', type=int, default=180)
    parser.add_argument('--load_epoch', type=int, default=0)

    args = parser.parse_args()

    n_in = args.n_in
    n_out = args.n_out
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    datasize = args.datasize
    ps = args.ps
    load_epoch = args.load_epoch

    ### TRAIN

    # data loader
    images = '/sequoia/data1/teboli/irc_nonblind/data/'
    datapath = '/sequoia/data2/teboli/irc_nonblind/data/training_uniform/'
    savepath = '/sequoia/data1/teboli/irc_nonblind/results/training_uniform/'
    savepath = os.path.join(savepath, 'net5_l1_aug_nout_%02d_nl_0.5to12.75' % (n_out))
    os.makedirs(savepath, exist_ok=True)
    modelpath = '/sequioa/data2/teboli/irc_non_blind/models/training_uniform/'
    modelpath = os.path.join(savepath, 'net5_l1_aug_nout_%02d_nin_%02d_ds_%05d_ps_%03d_nl_0.5to12.75_bs_%d' % (n_out, n_in, datasize, ps, batch_size))
    os.makedirs(modelpath, exist_ok=True)
    scorepath = '/sequioa/data1/teboli/irc_non_blind/models/training_uniform/'
    scorepath = os.path.join(savepath, 'net5_l1_aug_nout_%02d_nin_%02d_ds_%05d_ps_%03d_nl_0.5to12.75_bs_%d.npy' % (n_out, n_in, datasize, ps, batch_size))
    dt_tr = datasets.UniformTrainDataset(datapath, datasize, ps, train=True, transform=True)
    dt_va = datasets.UniformTrainDataset(datapath, datasize, ps, train=False)

    loader_tr = DataLoader(dt_tr, batch_size=batch_size, shuffle=True, num_workers=4)
    loader_va = DataLoader(dt_va, batch_size=batch_size, shuffle=False, num_workers=4)

    # model
    model = networks.IRC_Net_v5(n_out, n_in)
    criterion = loss.L1LossGreedy()
    if load_epoch > 0:
        filename = 'epoch_%03d.pt' % (load_epoch)
        state_dict_path = os.path.join(modelpath, filename)
        model.load_state_dict(torch.load(state_dict_path))

    # inverse filters
    k1 = model.weight[0].data
    d1 = kernels.compute_inverse_filter_basic(k1, 1e-2, 31).unsqueeze(0)
    k2 = model.weight[1].data
    d2 = kernels.compute_inverse_filter_basic(k2, 1e-2, 31).unsqueeze(0)

    # optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=n_epochs//2)
    model = model.to(device)
    criterion = criterion.to(device)
    d1 = d1.to(device)
    d2 = d2.to(device)

    # loop
    scores = np.zeros((4, 2*n_epochs))
    if load_epoch < n_epochs:
        for epoch in range(load_epoch, n_epochs, 1):
            print('### Epoch %03d ###' % (epoch+1))
            train_l1 = train_greedy(loader_tr, model, optimizer, criterion, epoch, d1, d2)
            print('   TR L1: %2.5f' % (train_l1))
            val_psnr, val_ssim, val_l1 = validate(loader_va, model, epoch, d1, d2)
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
            np.save(scorepath, scores)


    ### FINETUNE
    loadpath = os.path.join(modelpath, 'epoch_%03d.pt' % n_epochs)

    # model
    model = networks.IRC_Net_v5(n_out, n_in)
    state_dict = torch.load(loadpath, map_location='cpu')
    model.load_state_dict(state_dict)
    if load_epoch > n_epochs:
        filename = 'epoch_%03d.pt' % (load_epoch)
        state_dict_path = os.path.join(modelpath, filename)
        model.load_state_dict(torch.load(state_dict_path))
    criterion = nn.L1Loss()

    # optimizer
    lr = lr / 10
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=n_epochs//2, gamma=0.2)
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(max(n_epochs, load_epoch), 2*n_epochs):
        print('### Epoch %03d ###' % (epoch+1))
        train_l1 = train_finetune(loader_tr, model, optimizer, criterion, epoch, d1, d2)
        print('   TR L1: %2.5f' % (train_l1))
        val_psnr, val_ssim, val_l1 = validate(loader_va, model, epoch, d1, d2)
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
        np.save(scorepath, scores)
