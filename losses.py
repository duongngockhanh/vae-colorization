import torch

from config import args


def vae_loss(mu, logvar, pred, gt, lossweights, batchsize):
    """
    Return the loss values of the VAE model.
    """
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
    """
    Return the distribution coefficients of the GMM.
    """
    gmm_mu = gmm_params[..., : args["hiddensize"] * args["nmix"]]
    gmm_mu.contiguous()
    gmm_pi_activ = gmm_params[..., args["hiddensize"] * args["nmix"] :]
    gmm_pi_activ.contiguous()
    gmm_pi = F.softmax(gmm_pi_activ, dim=1)
    return gmm_mu, gmm_pi


def mdn_loss(gmm_params, mu, stddev, batchsize):
    """
    Calculates the loss by comparing two distribution
    - the predicted distribution of the MDN (given by gmm_mu and gmm_pi) with
    - the target distribution created by the Encoder block (given by mu and stddev).
    """
    gmm_mu, gmm_pi = get_gmm_coeffs(gmm_params)
    eps = torch.randn(stddev.size()).normal_().cuda()
    z = torch.add(mu, torch.mul(eps, stddev))
    z_flat = z.repeat(1, args["nmix"])
    z_flat = z_flat.reshape(batchsize * args["nmix"], args["hiddensize"])
    gmm_mu_flat = gmm_mu.reshape(batchsize * args["nmix"], args["hiddensize"])
    dist_all = torch.sqrt(torch.sum(torch.add(z_flat, gmm_mu_flat.mul(-1)).pow(2).mul(50), 1))
    dist_all = dist_all.reshape(batchsize, args["nmix"])
    dist_min, selectids = torch.min(dist_all, 1)
    gmm_pi_min = torch.gather(gmm_pi, 1, selectids.reshape(-1, 1))
    gmm_loss = torch.mean(torch.add(-1 * torch.log(gmm_pi_min + 1e-30), dist_min))
    gmm_loss_l2 = torch.mean(dist_min)
    return gmm_loss, gmm_loss_l2