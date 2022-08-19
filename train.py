#!/usr/bin/env python3
import torch
import os.path
import logging
import argparse
from collections import OrderedDict
from utils.data_loader import trainDataset
from utils import utils_logger
from utils import utils_image as util

from torch.utils.data import DataLoader
from model.phaseformer import build_model
from utils.util_loss import CharbonnierLoss
from utils.lr_schedule import create_multistep_lr_schedule

# test result dict
test_results = OrderedDict()
test_results['psnr'] = []
test_results['ssim'] = []
save_models_dir = './model_zoo'
trainset_path = './data/trainset/'
milestones = [100000, 200000, 300000, 400000]

def args_pharse():
    parser = argparse.ArgumentParser(description="PyTorch Transformer Training")
    parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
    parser.add_argument('--cuda', type=bool, default=True, help='True to use cuda')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs')
    parser.add_argument('--ckpt', type=int, default=0, help='load checkpoint')
    parser.add_argument('--downsample', type=int, default=4, help='downsample rate of 2^n')
    parser.add_argument('--patchsize', type=int, default=128, help='patch size to train')
    parser.add_argument('--name', type=str, default='phaseformer_T', help='model name')
    args = parser.parse_args()
    return args


def train_denoise():
    # ----------------- Preparation -----------------------
    args = args_pharse()
    modelname = "{}_d{}".format(args.name, args.downsample)
    logger_name = 'train_{}.log'.format(modelname)
    utils_logger.logger_info(logger_name, log_path=os.path.join('./logs', logger_name))
    logger = logging.getLogger(logger_name)
    trainset = trainDataset(root=trainset_path, patch_size=args.patchsize)
    loader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=4, pin_memory=True)
    steps = len(loader)

    lr = 1e-4 * (args.batchsize / 16)
    net = build_model(name=args.name, ds=args.downsample)
    optimizer = torch.optim.AdamW(
        [{'params': [p for p in net.parameters() if p.requires_grad], 'initial_lr': lr}],
        lr=lr, betas=(0.9, 0.99), weight_decay=5e-2)

    if args.cuda:
        net = net.cuda()
        device = 'cuda'
        torch.cuda.empty_cache()
    else:
        device = 'cpu'

    if args.ckpt != 0:
        net.load_state_dict(torch.load(
            os.path.join(save_models_dir, 'model_{}_{}.pth'.format(modelname, args.ckpt))))
        optimizer.load_state_dict(torch.load(
            os.path.join(save_models_dir, 'optim_{}_{}.pth'.format(modelname, args.ckpt))))
        scheduler = create_multistep_lr_schedule(optimizer, milestones=milestones, gamma=0.5, last_epoch=args.ckpt)
    else:
        scheduler = create_multistep_lr_schedule(optimizer, milestones=milestones, gamma=0.5, last_epoch=-1)

    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info("Total parameters: {:.2f}".format(params / 1e6))

    charbonnier_loss = CharbonnierLoss()  # objective function

    begin, end = args.ckpt + 1, args.epochs + 1
    for epoch in range(begin, end):
        for i, data_dict in enumerate(loader, 0):
            optimizer.zero_grad()
            L = data_dict['L'].to(device)
            H = data_dict['H'].to(device)
            N = data_dict['noise'].to(device)
            E = net.forward(L, N)
            loss = charbonnier_loss(E, H)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 100 == 0:
                e = util.tensor2uint(E)
                h = util.tensor2uint(H)
                psnr = util.calculate_psnr(e, h, border=0)
                logger.info("Epoch[{}/{}], Step[{}/{}], loss: {:.3f}, PSNR: {:.2f}"
                            .format(epoch, end - 1, i, steps, loss.item(), psnr))

        if epoch % 100 == 0:
            torch.save(net.state_dict(),
                       os.path.join(save_models_dir, 'model_{}_{}.pth'.format(modelname, epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(save_models_dir, 'optim_{}_{}.pth'.format(modelname, epoch)))


if __name__ == "__main__":
    train_denoise()
