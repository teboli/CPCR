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
        x, y, mag, ori = data
        x = x.to(device)
        y = y.to(device)
        mag = mag.to(device)
        ori = ori.to(device)
        ori = (90-ori).add(360).fmod(180)

        labels = utils.get_labels(mag, ori)
        ori = ori * np.pi / 180

        nl = 2.55
        nl = float(nl) / 255
        y += nl*torch.randn_like(y)
        y = y.clamp(0, 1)
        y.requires_grad_()

        optimizer.zero_grad()

        # pdb.set_trace()

        hat_x = model(y, mag, ori, labels, k1, k2, d1, d2)

        # pdb.set_trace()

        error = criterion(hat_x, x)
        error.backward()

        running_l1 += F.l1_loss(hat_x[-1], x).item()
        train_l1 += F.l1_loss(hat_x[-1], x).item()

        optimizer.step()

        # pdb.set_trace()

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
        x, y, mag, ori = data
        x = x.to(device)
        y = y.to(device)
        mag = mag.to(device)
        ori = ori.to(device)
        ori = (90-ori).add(360).fmod(180)

        labels = utils.get_labels(mag, ori)
        ori = ori * np.pi / 180

        nl = 2.55
        nl = float(nl) / 255
        y += nl*torch.randn_like(y)
        y = y.clamp(0, 1)
        y.requires_grad_()

        optimizer.zero_grad()

        hat_x = model(y, mag, ori, labels, k1, k2, d1, d2)

        error = criterion(hat_x, x)
        error.backward()

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

    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(loader)):
            x, y, mag, ori  = data
            x = x.to(device)
            y = y.to(device)
            mag = mag.to(device)
            ori = ori.to(device)
            ori = (90-ori).add(360).fmod(180)

            labels = utils.get_labels(mag, ori)
            ori = ori * np.pi / 180

            nl = 2.55 / 255
            y += nl*torch.randn_like(y)
            y = y.clamp(0, 1)

            hat_x = model(y, mag, ori, labels, k1, k2, d1, d2)

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

    args = parser.parse_args()

    n_in = args.n_in
    n_out = args.n_out
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    datasize = args.datasize
    ps = args.ps

    ### TRAIN

    # data loader
    images = '/sequoia/data1/teboli/irc_nonblind/data/'
    datapath = '/sequoia/data1/teboli/irc_nonblind/data/training_nonuniform/'
    savepath = '/sequoia/data1/teboli/irc_nonblind/results/training_nonuniform/'
    savepath = os.path.join(savepath, 'net4_nu_l1_aug_nout_%02d_nl_2.55' % (n_out))
    os.makedirs(savepath, exist_ok=True)
    modelpath = '/sequioa/data2/teboli/irc_non_blind/models/training_nonuniform/'
    modelpath = os.path.join(savepath, 'net4_nu_l1_aug_nout_%02d_nin_%02d_ds_%05d_ps_%03d_nl_2.55_bs_%d' % (n_out, n_in, datasize, ps, batch_size))
    os.makedirs(modelpath, exist_ok=True)
    scorepath = '/sequioa/data1/teboli/irc_non_blind/models/training_nonuniform/'
    scorepath = os.path.join(savepath, 'net4_nu_l1_aug_nout_%02d_nin_%02d_ds_%05d_ps_%03d_nl_2.55_bs_%d.npy' % (n_out, n_in, datasize, ps, batch_size))
    dt_tr = datasets.NonUniformTrainDataset(datapath, datasize, ps, train=True, transform=True)
    dt_va = datasets.NonUniformTrainDataset(datapath, datasize, ps, train=False)

    loader_tr = DataLoader(dt_tr, batch_size=batch_size, shuffle=True, num_workers=4)
    loader_va = DataLoader(dt_va, batch_size=batch_size, shuffle=False, num_workers=4)

    # model
    fts = torch.load('/sequoia/data1/teboli/irc_nonblind/data/kernels/kers_grad_0.pt')
    weights = fts[:, 0].unsqueeze(1)

    model = networks.IRC_Net_NU_v4(weights, n_out, n_in)
    criterion = loss.L1LossGreedy()

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
    for epoch in range(n_epochs):
        print('### Epoch %03d ###' % (epoch+1))
        train_l1 = train_greedy(loader_tr, model, optimizer, criterion, epoch, d1, d2)
        print('   TR L1: %2.5f' % (train_l1))
        val_psnr, val_ssim, val_l1 = validate(loader_va, model, epoch, d1, d2)
        print('   VA L1: %2.5f / PSNR: %2.2f / SSIM: %2.3f' % (val_l1, val_psnr, val_ssim))

        # scheduler.step()

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
    model = networks.IRC_Net_NU_v4(weights, n_out, n_in)
    state_dict = torch.load(loadpath, map_location='cpu')
    model.load_state_dict(state_dict)
    criterion = nn.L1Loss()

    # optimizer
    # lr = lr / 10
    lr = 1e-5
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=n_epochs//2, gamma=0.2)
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(n_epochs, 2*n_epochs):
        print('### Epoch %03d ###' % (epoch+1))
        train_l1 = train_finetune(loader_tr, model, optimizer, criterion, epoch, d1, d2)
        print('   TR L1: %2.5f' % (train_l1))
        val_psnr, val_ssim, val_l1 = validate(loader_va, model, epoch, d1, d2)
        print('   VA L1: %2.5f / PSNR: %2.2f / SSIM: %2.3f' % (val_l1, val_psnr, val_ssim))

        # scheduler.step()

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
