#!/usr/bin/env python3
import os
import os.path
import logging
import argparse
from collections import OrderedDict

from utils.data_loader import *
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_model
from model.phaseformer import build_model
from utils.crop_ import *
from utils import utils_image as util

save_models_dir = './model_zoo'
testset_path = './data/testset/'
result_path = './results'


def args_pharse():
    parser = argparse.ArgumentParser(description="PyTorch Transformer Training")
    parser.add_argument('--ckpt', type=int, default=1100, help='load checkpoint')
    parser.add_argument('--sigma', type=int, default=50, help='Gaussian noise level')
    parser.add_argument('--dataset', type=str, default='CBSD68', help='Gaussian noise level')
    parser.add_argument('--name', type=str, default='phaseformer_S', help='model name')
    parser.add_argument('--downsample', type=int, default=4, help='downsample rate of 2^n')
    parser.add_argument('--mode', type=int, default=4, help='test mode, 3: x8, 4: split x8')
    parser.add_argument('--save_results', action='store_true', help='To save the result images')
    args = parser.parse_args()
    return args


def test_denoise():
    # ----------------- Preparation -----------------------
    border = 0  # shave border to calculate PSNR and SSIM
    args = args_pharse()
    modelname = f"{args.name}_d{args.downsample}"
    logger_name = f'_{args.dataset}_{args.ckpt}_{modelname}.log'
    utils_logger.logger_info(logger_name, log_path=os.path.join('./logs', logger_name))
    logger = logging.getLogger(logger_name)
    data_path = os.path.join(testset_path, args.dataset)
    save_img_dir = os.path.join(result_path, f"{args.dataset}_denoise")
    util.mkdir(save_img_dir)

    if not os.path.exists(data_path):
        raise FileNotFoundError

    net = build_model(name=args.name, ds=args.downsample)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
        device = 'cuda'
        torch.cuda.empty_cache()
    else:
        device = 'cpu'
    # ----------------- model -----------------------
    net.load_state_dict(torch.load(
        os.path.join(save_models_dir, f'model_{modelname}_{args.ckpt}.pth')))
    net.eval()
    for k, v in net.named_parameters():
        v.requires_grad = False

    # test result dict
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    testset = testDataset(root=data_path, sigma=args.sigma)
    loader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    for i, data_dict in enumerate(loader, 0):
        L = data_dict['L'].to(device)
        H = data_dict['H'].to(device)
        N = data_dict['noise'].to(device)
        img_name = data_dict['name'][0]

        y = utils_model.test_mode(model=net, L=L, N=N, mode=args.mode)

        # ------------- PSNR and SSIM -------------------
        img_E = util.tensor2uint(y)
        img_H = util.tensor2uint(H)
        img_L = util.tensor2uint(L)

        if args.save_results:
            util.imsave(np.concatenate([img_L, img_E, img_H], axis=1),
                        os.path.join(save_img_dir, img_name + modelname + '_LEH.png'))

        psnr = util.calculate_psnr(img_E, img_H, border=border)
        ssim = util.calculate_ssim(img_E, img_H, border=border)

        psnr_L = util.calculate_psnr(img_L, img_H, border=border)
        ssim_L = util.calculate_ssim(img_L, img_H, border=border)
        logger.info('{}: PSNR_L: {:.2f} dB; PSNR_E: {:.2f} dB; SSIM_L: {:.2f}; SSIM_E: {:.2f}'.format(img_name, psnr_L, psnr, ssim_L, ssim))
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)

    avg_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    avg_ssim = sum(test_results['ssim']) / len(test_results['ssim'])

    logger.info('model: {}, sigma: {}, Average PSNR/SSIM(RGB) - PSNR: {:.2f} dB; SSIM: {:.2f}'
                .format(modelname, args.sigma, avg_psnr, avg_ssim))


if __name__ == '__main__':
    test_denoise()
