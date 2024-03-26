from __future__ import print_function

import argparse
import os

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from colordata import colordata
from vae import VAE
from mdn import MDN

parser = argparse.ArgumentParser(description="PyTorch Diverse Colorization")
parser.add_argument("dataset_key", help="Dataset")
parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU device id")
parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("-b", "--batchsize", type=int, default=32, help="Batch size")
parser.add_argument("-z", "--hiddensize", type=int, default=64, help="Latent vector dimension")
parser.add_argument("-n", "--nthreads", type=int, default=4, help="Data loader threads")
parser.add_argument("-em", "--epochs_mdn", type=int, default=1, help="Number of epochs for MDN")
parser.add_argument("-m", "--nmix", type=int, default=8, help="Number of diverse colorization (or output gmm components)")
parser.add_argument('-lg', '--logstep', type=int, default=100, help='Interval to log data')
args = parser.parse_args()


def get_dirpaths(args):
    if args.dataset_key == "lfw":
        out_dir = "data/output/lfw"
        listdir = "data/imglist/lfw"
        featslistdir = "data/featslist/lfw"
    else:
        raise NameError("[ERROR] Incorrect key: %s" % (args.dataset_key))
    return out_dir, listdir, featslistdir


def vae_loss(mu, logvar, pred, gt, lossweights, batchsize):
    kl_element = torch.add(torch.add(torch.add(mu.pow(2), logvar.exp()), -1), logvar.mul(-1))
    kl_loss = torch.sum(kl_element).mul(0.5)
    gt = gt.reshape(-1, 64 * 64 * 2)
    pred = pred.reshape(-1, 64 * 64 * 2)
    recon_element = torch.sqrt(torch.sum(torch.mul(torch.add(gt, pred.mul(-1)).pow(2), lossweights), 1))
    recon_loss = torch.sum(recon_element).mul(1.0 / (batchsize))

    recon_element_l2 = torch.sqrt(torch.sum(torch.add(gt, pred.mul(-1)).pow(2), 1))
    recon_loss_l2 = torch.sum(recon_element_l2).mul(1.0 / (batchsize))

    return kl_loss, recon_loss, recon_loss_l2


def get_gmm_coeffs(gmm_params):
    gmm_mu = gmm_params[..., : args.hiddensize * args.nmix]
    gmm_mu.contiguous()
    gmm_pi_activ = gmm_params[..., args.hiddensize * args.nmix :]
    gmm_pi_activ.contiguous()
    gmm_pi = F.softmax(gmm_pi_activ, dim=1)
    return gmm_mu, gmm_pi


def mdn_loss(gmm_params, mu, stddev, batchsize):
    gmm_mu, gmm_pi = get_gmm_coeffs(gmm_params)
    eps = torch.randn(stddev.size()).normal_().cuda()
    z = torch.add(mu, torch.mul(eps, stddev))
    z_flat = z.repeat(1, args.nmix)
    z_flat = z_flat.reshape(batchsize * args.nmix, args.hiddensize)
    gmm_mu_flat = gmm_mu.reshape(batchsize * args.nmix, args.hiddensize)
    dist_all = torch.sqrt(torch.sum(torch.add(z_flat, gmm_mu_flat.mul(-1)).pow(2).mul(50), 1))
    dist_all = dist_all.reshape(batchsize, args.nmix)
    dist_min, selectids = torch.min(dist_all, 1)
    gmm_pi_min = torch.gather(gmm_pi, 1, selectids.reshape(-1, 1))
    gmm_loss = torch.mean(torch.add(-1 * torch.log(gmm_pi_min + 1e-30), dist_min))
    gmm_loss_l2 = torch.mean(dist_min)
    return gmm_loss, gmm_loss_l2


def test_vae(model):
    model.eval()

    out_dir, listdir, featslistdir = get_dirpaths(args)
    batchsize = args.batchsize
    hiddensize = args.hiddensize
    nmix = args.nmix

    data = colordata(
        os.path.join(out_dir, "images"),
        listdir=listdir,
        featslistdir=featslistdir,
        split="test",
    )
    nbatches = np.int_(np.floor(data.img_num / batchsize))
    data_loader = DataLoader(
        dataset=data,
        num_workers=args.nthreads,
        batch_size=batchsize,
        shuffle=False,
        drop_last=True,
    )

    test_loss = 0.0
    for batch_idx, (
        batch,
        batch_recon_const,
        batch_weights,
        batch_recon_const_outres,
        _,
    ) in tqdm(enumerate(data_loader), total=nbatches):

        input_color = batch.cuda()
        lossweights = batch_weights.cuda()
        lossweights = lossweights.reshape(batchsize, -1)
        input_greylevel = batch_recon_const.cuda()
        z = torch.randn(batchsize, hiddensize)

        mu, logvar, color_out = model(input_color, input_greylevel, z)
        _, _, recon_loss_l2 = vae_loss(
            mu, logvar, color_out, input_color, lossweights, batchsize
        )
        test_loss = test_loss + recon_loss_l2.item()

    test_loss = (test_loss * 1.0) / nbatches
    model.train()

    return test_loss


def train_vae():
    out_dir, listdir, featslistdir = get_dirpaths(args)
    batchsize = args.batchsize
    hiddensize = args.hiddensize
    nmix = args.nmix
    nepochs = args.epochs

    data = colordata(
        os.path.join(out_dir, "images"),
        listdir=listdir,
        featslistdir=featslistdir,
        split="train",
    )
    nbatches = np.int_(np.floor(data.img_num / batchsize))
    data_loader = DataLoader(
        dataset=data,
        num_workers=args.nthreads,
        batch_size=batchsize,
        shuffle=True,
        drop_last=True,
    )

    model = VAE()
    model.cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    itr_idx = 0
    for epochs in range(nepochs):
        train_loss = 0.0

        for batch_idx, (
            batch,
            batch_recon_const,
            batch_weights,
            batch_recon_const_outres,
            _,
        ) in tqdm(enumerate(data_loader), total=nbatches):

            input_color = batch.cuda()
            lossweights = batch_weights.cuda()
            lossweights = lossweights.reshape(batchsize, -1)
            input_greylevel = batch_recon_const.cuda()
            z = torch.randn(batchsize, hiddensize)

            optimizer.zero_grad()
            mu, logvar, color_out = model(input_color, input_greylevel, z)
            kl_loss, recon_loss, recon_loss_l2 = vae_loss(
                mu, logvar, color_out, input_color, lossweights, batchsize
            )
            loss = kl_loss.mul(1e-2) + recon_loss
            recon_loss_l2.detach()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + recon_loss_l2.item()

            if batch_idx % args.logstep == 0:
                data.saveoutput_gt(
                    color_out.cpu().data.numpy(),
                    batch.numpy(),
                    "train_%05d_%05d" % (epochs, batch_idx),
                    batchsize,
                    net_recon_const=batch_recon_const_outres.numpy(),
                )

        train_loss = (train_loss * 1.0) / (nbatches)
        print("[DEBUG] VAE Train Loss, epoch %d has loss %f" % (epochs, train_loss))
        test_loss = test_vae(model)
        print("[DEBUG] VAE Test Loss, epoch %d has loss %f" % (epochs, test_loss))
        torch.save(model.state_dict(), "%s/models/model_vae.pth" % (out_dir))

    print("Complete VAE training")


def train_mdn():
    out_dir, listdir, featslistdir = get_dirpaths(args)
    batchsize = args.batchsize
    hiddensize = args.hiddensize
    nmix = args.nmix
    nepochs = args.epochs_mdn

    data = colordata(
        os.path.join(out_dir, "images"),
        listdir=listdir,
        featslistdir=featslistdir,
        split="train",
    )

    nbatches = np.int_(np.floor(data.img_num / batchsize))

    data_loader = DataLoader(
        dataset=data,
        num_workers=args.nthreads,
        batch_size=batchsize,
        shuffle=True,
        drop_last=True,
    )

    model_vae = VAE()
    model_vae.cuda()
    model_vae.load_state_dict(torch.load("%s/models/model_vae.pth" % (out_dir)))
    model_vae.eval()

    model_mdn = MDN()
    model_mdn.cuda()
    model_mdn.train()

    optimizer = optim.Adam(model_mdn.parameters(), lr=1e-3)

    itr_idx = 0
    for epochs_mdn in range(nepochs):
        train_loss = 0.0

        for batch_idx, (
            batch,
            batch_recon_const,
            batch_weights,
            _,
            batch_feats,
        ) in tqdm(enumerate(data_loader), total=nbatches):

            input_color = batch.cuda()
            input_greylevel = batch_recon_const.cuda()
            input_feats = batch_feats.cuda()
            z = torch.randn(batchsize, hiddensize)

            optimizer.zero_grad()

            mu, logvar, _ = model_vae(input_color, input_greylevel, z)
            mdn_gmm_params = model_mdn(input_feats)

            loss, loss_l2 = mdn_loss(
                mdn_gmm_params, mu, torch.sqrt(torch.exp(logvar)), batchsize
            )
            loss.backward()

            optimizer.step()
            train_loss = train_loss + loss.item()

        train_loss = (train_loss * 1.0) / (nbatches)
        print("[DEBUG] Training MDN, epoch %d has loss %f" % (epochs_mdn, train_loss))
        torch.save(model_mdn.state_dict(), "%s/models/model_mdn.pth" % (out_dir))

    print("Complete MDN training")


def divcolor():
    out_dir, listdir, featslistdir = get_dirpaths(args)
    batchsize = args.batchsize
    hiddensize = args.hiddensize
    nmix = args.nmix

    data = colordata(
        os.path.join(out_dir, "images"),
        listdir=listdir,
        featslistdir=featslistdir,
        split="test",
    )

    nbatches = np.int_(np.floor(data.img_num / batchsize))

    data_loader = DataLoader(
        dataset=data,
        num_workers=args.nthreads,
        batch_size=batchsize,
        shuffle=True,
        drop_last=True,
    )

    model_vae = VAE()
    model_vae.cuda()
    model_vae.load_state_dict(torch.load("%s/models/model_vae.pth" % (out_dir)))
    model_vae.eval()

    model_mdn = MDN()
    model_mdn.cuda()
    model_mdn.load_state_dict(torch.load("%s/models/model_mdn.pth" % (out_dir)))
    model_mdn.eval()

    for batch_idx, (
        batch,
        batch_recon_const,
        batch_weights,
        batch_recon_const_outres,
        batch_feats,
    ) in tqdm(enumerate(data_loader), total=nbatches):

        input_feats = batch_feats.cuda()

        mdn_gmm_params = model_mdn(input_feats)
        gmm_mu, gmm_pi = get_gmm_coeffs(mdn_gmm_params)
        gmm_pi = gmm_pi.reshape(-1, 1)
        gmm_mu = gmm_mu.reshape(-1, hiddensize)

        for j in range(batchsize):
            batch_j = np.tile(batch[j, ...].numpy(), (batchsize, 1, 1, 1))
            batch_recon_const_j = np.tile(
                batch_recon_const[j, ...].numpy(), (batchsize, 1, 1, 1)
            )
            batch_recon_const_outres_j = np.tile(
                batch_recon_const_outres[j, ...].numpy(), (batchsize, 1, 1, 1)
            )

            input_color = torch.from_numpy(batch_j).cuda()
            input_greylevel = torch.from_numpy(batch_recon_const_j).cuda()

            curr_mu = gmm_mu[j * nmix : (j + 1) * nmix, :]
            orderid = np.argsort(
                gmm_pi[j * nmix : (j + 1) * nmix, 0].cpu().data.numpy().reshape(-1)
            )

            z = curr_mu.repeat(int((batchsize * 1.0) / nmix), 1)

            _, _, color_out = model_vae(input_color, input_greylevel, z)

            data.saveoutput_gt(
                color_out.cpu().data.numpy()[orderid, ...],
                batch_j[orderid, ...],
                "divcolor_%05d_%05d" % (batch_idx, j),
                nmix,
                net_recon_const=batch_recon_const_outres_j[orderid, ...],
            )

    print("Complete inference")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_vae()
    train_mdn()
    divcolor()
