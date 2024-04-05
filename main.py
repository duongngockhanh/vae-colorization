import os

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import ColorDataset
from vae import VAE
from mdn import MDN
from config import args
from losses import vae_loss, get_gmm_coeffs, mdn_loss

def get_dirpaths(args):
    if args.dataset_key == "lfw":
        out_dir = "data/output/lfw"
        listdir = "data/imglist/lfw"
        featslistdir = "data/featslist/lfw"
    else:
        raise NameError("[ERROR] Incorrect key: %s" % (args.dataset_key))
    return out_dir, listdir, featslistdir


def test_vae(model):
    model.eval()

    # Load hyperparameters
    out_dir, listdir, featslistdir = get_dirpaths(args)
    batchsize = args["batchsize"]
    hiddensize = args["hiddensize"]
    nmix = args["nmix"]

    # Create DataLoader
    data = ColorDataset(os.path.join(out_dir, "images"), listdir, featslistdir, split="test")
    nbatches = np.int_(np.floor(data.img_num / batchsize))
    data_loader = DataLoader(dataset=data, num_workers=args["nthreads"], batch_size=batchsize, shuffle=False, drop_last=True)

    # Eval
    test_loss = 0.0
    for batch_idx, (batch, batch_recon_const, batch_weights, batch_recon_const_outres, _) in tqdm(enumerate(data_loader), total=nbatches):
        input_color = batch.cuda()
        lossweights = batch_weights.cuda()
        lossweights = lossweights.reshape(batchsize, -1)
        input_greylevel = batch_recon_const.cuda()
        z = torch.randn(batchsize, hiddensize)

        mu, logvar, color_out = model(input_color, input_greylevel, z)
        _, _, recon_loss_l2 = vae_loss(mu, logvar, color_out, input_color, lossweights, batchsize)
        test_loss = test_loss + recon_loss_l2.item()

    test_loss = (test_loss * 1.0) / nbatches
    model.train()
    return test_loss


def train_vae():
    # Load hyperparameters
    out_dir, listdir, featslistdir = get_dirpaths(args)
    batchsize = args["batchsize"]
    hiddensize = args["hiddensize"]
    nmix = args["nmix"]
    nepochs = args["epochs"]

    # Create DataLoader
    data = ColorDataset(os.path.join(out_dir, "images"), listdir, featslistdir, split="train")
    nbatches = np.int_(np.floor(data.img_num / batchsize))
    data_loader = DataLoader(dataset=data, num_workers=args["nthreads"], batch_size=batchsize, shuffle=True, drop_last=True)

    # Initialize VAE model
    model = VAE()
    model.cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    # Train
    for epochs in range(nepochs):
        train_loss = 0.0

        for batch_idx, (batch, batch_recon_const, batch_weights, batch_recon_const_outres, _) in tqdm(enumerate(data_loader), total=nbatches):
            input_color = batch.cuda()
            lossweights = batch_weights.cuda()
            lossweights = lossweights.reshape(batchsize, -1)
            input_greylevel = batch_recon_const.cuda()
            z = torch.randn(batchsize, hiddensize)

            optimizer.zero_grad()
            mu, logvar, color_out = model(input_color, input_greylevel, z)
            kl_loss, recon_loss, recon_loss_l2 = vae_loss(mu, logvar, color_out, input_color, lossweights, batchsize)
            loss = kl_loss.mul(1e-2) + recon_loss
            recon_loss_l2.detach()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + recon_loss_l2.item()

            if batch_idx % args["logstep"] == 0:
                data.saveoutput_gt(
                    color_out.cpu().data.numpy(),
                    batch.numpy(),
                    "train_%05d_%05d" % (epochs, batch_idx),
                    batchsize,
                    net_recon_const=batch_recon_const_outres.numpy()
                )

        train_loss = (train_loss * 1.0) / (nbatches)
        test_loss = test_vae(model)
        print(f"End of epoch {epochs:3d} | Train Loss {train_loss:8.3f} | Test Loss {test_loss:8.3f} ")

        # Save VAE model
        torch.save(model.state_dict(), "%s/models/model_vae.pth" % (out_dir))

    print("Complete VAE training")


def test_mdn(model_vae, model_mdn):
    # Load hyperparameters
    out_dir, listdir, featslistdir = get_dirpaths(args)
    batchsize = args["batchsize"]
    hiddensize = args["hiddensize"]
    nmix = args["nmix"]

    # Create DataLoader
    data = ColorDataset(os.path.join(out_dir, "images"), listdir, featslistdir, split="test")
    nbatches = np.int_(np.floor(data.img_num / batchsize))
    data_loader = DataLoader(dataset=data, num_workers=args["nthreads"], batch_size=batchsize, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model_mdn.parameters(), lr=1e-3)

    # Eval
    model_vae.eval()
    model_mdn.eval()
    itr_idx = 0
    test_loss = 0.0

    for batch_idx, (batch, batch_recon_const, batch_weights, _, batch_feats) in tqdm(enumerate(data_loader), total=nbatches):
        input_color = batch.cuda()
        input_greylevel = batch_recon_const.cuda()
        input_feats = batch_feats.cuda()
        z = torch.randn(batchsize, hiddensize)
        optimizer.zero_grad()

        # Get the parameters of the posterior distribution
        mu, logvar, _ = model_vae(input_color, input_greylevel, z)

        # Get the GMM vector
        mdn_gmm_params = model_mdn(input_feats)

        # Compare 2 distributions
        loss, _ = mdn_loss(mdn_gmm_params, mu, torch.sqrt(torch.exp(logvar)), batchsize)


        test_loss = test_loss + loss.item()

    test_loss = (test_loss * 1.0) / (nbatches)
    model_vae.train()
    return test_loss


def train_mdn():
    # Load hyperparameters
    out_dir, listdir, featslistdir = get_dirpaths(args)
    batchsize = args["batchsize"]
    hiddensize = args["hiddensize"]
    nmix = args["nmix"]
    nepochs = args["epochs_mdn"]

    # Create DataLoader
    data = ColorDataset(os.path.join(out_dir, "images"), listdir, featslistdir, split="train")
    nbatches = np.int_(np.floor(data.img_num / batchsize))
    data_loader = DataLoader(dataset=data, num_workers=args["nthreads"], batch_size=batchsize, shuffle=True, drop_last=True)

    # Initialize VAE model
    model_vae = VAE()
    model_vae.cuda()
    model_vae.load_state_dict(torch.load("%s/models/model_vae.pth" % (out_dir)))
    model_vae.eval()

    # Initialize MDN model
    model_mdn = MDN()
    model_mdn.cuda()
    model_mdn.train()

    optimizer = optim.Adam(model_mdn.parameters(), lr=1e-3)

    # Train
    for epochs_mdn in range(nepochs):
        train_loss = 0.0

        for _, (batch, batch_recon_const, _, _, batch_feats) in tqdm(enumerate(data_loader), total=nbatches):
            input_color = batch.cuda()
            input_greylevel = batch_recon_const.cuda()
            input_feats = batch_feats.cuda()
            z = torch.randn(batchsize, hiddensize)
            optimizer.zero_grad()

            # Get the parameters of the posterior distribution
            mu, logvar, _ = model_vae(input_color, input_greylevel, z)

            # Get the GMM vector
            mdn_gmm_params = model_mdn(input_feats)

            # Compare 2 distributions
            loss, _ = mdn_loss(mdn_gmm_params, mu, torch.sqrt(torch.exp(logvar)), batchsize)

            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()

        train_loss = (train_loss * 1.0) / (nbatches)
        test_loss = test_mdn(model_vae, model_mdn)
        print(f"\nEnd of epoch {epochs_mdn:3d} | Train Loss {train_loss:8.3f} |  Test Loss {test_loss:8.3f}")

        # Save MDN model
        torch.save(model_mdn.state_dict(), "%s/models_mdn/model_mdn.pth" % (out_dir))

    print("Complete MDN training")


def inference(vae_ckpt=None, mdn_ckpt=None):
    # Load hyperparameters
    out_dir, listdir, featslistdir = get_dirpaths(args)
    batchsize = args["batchsize"]
    hiddensize = args["hiddensize"]
    nmix = args["nmix"]

    # Create DataLoader
    data = ColorDataset(os.path.join(out_dir, "images"), listdir, featslistdir, split="test")
    nbatches = np.int_(np.floor(data.img_num / batchsize))
    data_loader = DataLoader(dataset=data, num_workers=args["nthreads"], batch_size=batchsize, shuffle=False, drop_last=True)

    # Initialize VAE model
    model_vae = VAE()
    model_vae.cuda()
    if vae_ckpt:
        model_vae.load_state_dict(torch.load(vae_ckpt))
    else:
        model_vae.load_state_dict(torch.load("%s/models/model_vae.pth" % (out_dir)))
    model_vae.eval()

    # Initialize MDN model
    model_mdn = MDN()
    model_mdn.cuda()
    if mdn_ckpt:
        model_mdn.load_state_dict(torch.load(mdn_ckpt))
    else:
        model_mdn.load_state_dict(torch.load("%s/models_mdn/model_mdn.pth" % (out_dir)))
    model_mdn.eval()

    # Infer
    for batch_idx, (batch, batch_recon_const, batch_weights, batch_recon_const_outres, batch_feats) in tqdm(enumerate(data_loader), total=nbatches):
        input_feats = batch_feats.cuda()

        # Get GMM parameters
        mdn_gmm_params = model_mdn(input_feats)
        gmm_mu, gmm_pi = get_gmm_coeffs(mdn_gmm_params)
        gmm_pi = gmm_pi.reshape(-1, 1)
        gmm_mu = gmm_mu.reshape(-1, hiddensize)

        for j in range(batchsize):
            batch_j = np.tile(batch[j, ...].numpy(), (batchsize, 1, 1, 1))
            batch_recon_const_j = np.tile(batch_recon_const[j, ...].numpy(), (batchsize, 1, 1, 1))
            batch_recon_const_outres_j = np.tile(batch_recon_const_outres[j, ...].numpy(), (batchsize, 1, 1, 1))

            input_color = torch.from_numpy(batch_j).cuda()
            input_greylevel = torch.from_numpy(batch_recon_const_j).cuda()

            # Get mean from GMM
            curr_mu = gmm_mu[j * nmix : (j + 1) * nmix, :]
            orderid = np.argsort(gmm_pi[j * nmix : (j + 1) * nmix, 0].cpu().data.numpy().reshape(-1))

            # Sample from GMM
            z = curr_mu.repeat(int((batchsize * 1.0) / nmix), 1)

            # Predict color
            _, _, color_out = model_vae(input_color, input_greylevel, z)

            data.saveoutput_gt(
                color_out.cpu().data.numpy()[orderid, ...],
                batch_j[orderid, ...],
                "divcolor_%05d_%05d" % (batch_idx, j),
                nmix,
                net_recon_const=batch_recon_const_outres_j[orderid, ...],
            )

    print("\nComplete inference. The results are saved in data/output/lfw/images.")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_vae()
    train_mdn()
    inference()
