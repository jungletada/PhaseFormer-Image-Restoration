#!/usr/bin/env python3
import os.path
import cv2
import logging
import torch
import numpy as np
import argparse
import hdf5storage
from scipy import ndimage
from collections import OrderedDict

from model.phaseformer import build_model
from utils import utils_logger
from utils import utils_model
from utils import utils_pnp as pnp
from utils import utils_sisr as sisr
from utils import utils_image as util
import warnings
warnings.filterwarnings("ignore")

save_models_dir = './model_zoo'
testset_path = './data/testset'
result_path = './results'
kernel_path = './kernels'


def args_pharse():
    parser = argparse.ArgumentParser(description="Test for image deblurring")
    parser.add_argument('--ckpt', type=int, default=1100, help='load checkpoint')
    parser.add_argument('--sigma', type=float, default=7.65, help='Gaussian noise level')
    parser.add_argument('--dataset', type=str, default='Set14', help='Gaussian noise level')
    parser.add_argument('--name', type=str, default='phaseformer_S', help='model name')
    parser.add_argument('--downsample', type=int, default=4, help='downsample rate of 2^n')
    parser.add_argument('--mode', type=int, default=2, help='test mode, 3: x8, 4: split x8')
    parser.add_argument('--augment', action='store_true', help='to use x8 augmentation')
    parser.add_argument('--iter', type=int, default=16, help='number of iterations')
    parser.add_argument('--show_iter_steps', action='store_true', help='To show intermediate steps')
    parser.add_argument('--save_results', action='store_true', help='To save the result images')
    args = parser.parse_args()
    return args


def main():
    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    args = args_pharse()
    noise_level_img = args.sigma / 255.0  # default: 0, noise level for LR image
    noise_level_model = noise_level_img  # noise level of model, default 0
    x8 = args.augment  # default: False, x8 to boost performance
    iter_num = args.iter  # number of iterations
    modelSigma1 = 8.29
    modelSigma2 = args.sigma
    sf = 1 # scale factor
    show_iter_steps = args.show_iter_steps
    # save_L = False  # save LR image
    # save_E = False  # save estimated image
    show_img = False  # default: False
    save_LEH = args.save_results  # save zoomed LR, E and H images
    border = 0

    # --------------------------------
    # load kernel
    # --------------------------------
    kernel_dict = {'Levin': 'Levin09.mat', '12': 'kernels_12.mat'}
    kernels = hdf5storage.loadmat(os.path.join(kernel_path, kernel_dict['Levin']))['kernels']
    result_name = f"{args.dataset}_deblur"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------
    L_path = os.path.join(testset_path, args.dataset)  # L_path, for Low-quality images
    E_path = os.path.join(result_path, result_name)  # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------
    modelname = f"{args.name}_d{args.downsample}"
    model = build_model(name=args.name, ds=args.downsample)
    model.load_state_dict(torch.load(
        os.path.join(save_models_dir, f'model_{modelname}_{args.ckpt}.pth')))
    model.eval()

    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    logger.info(
        'model_name:{}, image sigma:{:.2f}%, model sigma:{:.2f}%'
            .format(modelname, noise_level_img*100, noise_level_model*100))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    test_results_ave = OrderedDict()
    test_results_ave['psnr'] = []  # record average PSNR for each kernel

    for k_index in range(kernels.shape[1]):
        logger.info('-------k:{:>2d} ---------'.format(k_index))
        test_results = OrderedDict()
        test_results['psnr'] = []
        k = kernels[0, k_index].astype(np.float64)
        util.imshow(k) if show_img else None
        logger.info(f'----kernel sizeL{k.shape[1:]}')
        for idx, img in enumerate(L_paths):
            # --------------------------------
            # (1) get img_L
            # --------------------------------
            img_name, ext = os.path.splitext(os.path.basename(img))
            img_H = util.imread_uint(img, n_channels=3)
            img_H = util.modcrop(img_H, 8)  # mod crop

            img_L = ndimage.filters.convolve(img_H, np.expand_dims(k, axis=2), mode='wrap')
            util.imshow(img_L) if show_img else None
            img_L = util.uint2single(img_L)

            np.random.seed(seed=6)  # for reproducibility
            img_L += np.random.normal(0, noise_level_img, img_L.shape)  # add AWGN

            # --------------------------------
            # (2) get rhos and sigmas
            # --------------------------------
            rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255/255., noise_level_img), iter_num=iter_num, lambda_=0.23,
                                             modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
            rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

            # --------------------------------
            # (3) initialize x, and pre-calculation
            # --------------------------------
            x = util.single2tensor4(img_L).to(device)
            img_L_tensor, k_tensor = util.single2tensor4(img_L), util.single2tensor4(np.expand_dims(k, 2))
            [k_tensor, img_L_tensor] = util.todevice([k_tensor, img_L_tensor], device)
            FB, FBC, F2B, FBFy = sisr.pre_calculate(img_L_tensor, k_tensor, sf)
            H = util.uint2tensor4(img_H)
            # --------------------------------
            # (4) main iterations
            # --------------------------------
            for i in range(iter_num):
                # --------------------------------
                # step 1, FFT
                # --------------------------------
                tau = rhos[i].float().repeat(1, 1, 1, 1)
                x = sisr.data_solution(x, FB, FBC, F2B, FBFy, tau, sf)
                # --------------------------------
                # step 2, denoiser
                # --------------------------------
                if x8:
                    x = util.augment_img_tensor4(x, i % 8)
                    H = util.augment_img_tensor4(H, i % 8)
                N = sigmas[i].float().repeat(1, 1, x.shape[2], x.shape[3])
                x = utils_model.test_mode(model, L=x, N=N, mode=2, refield=32, min_size=256, modulo=16)
                
                if args.show_iter_steps:
                    img_inter = util.tensor2uint(x)
                    img_inter_H = util.tensor2uint(H)
                    # util.imsave(img_inter, os.path.join(E_path, f"{img_name}_k{k_index}_{modelname}_iter.{i}.png"))
                    psnr = util.calculate_psnr(img_inter, img_inter_H, border=border)
                    # print("iters: {}, psnr={:.2f}".format(i, psnr))

                if x8:
                    if i % 8 == 3 or i % 8 == 5:
                        x = util.augment_img_tensor4(x, 8 - i % 8)
                        H = util.augment_img_tensor4(H, 8 - i % 8)
                    else:
                        x = util.augment_img_tensor4(x, i % 8)
                        H = util.augment_img_tensor4(H, i % 8)
            # --------------------------------
            # (3) img_E 
            # --------------------------------
            img_E = util.tensor2uint(x)
            # --------------------------------
            # (4) img_LEH
            # --------------------------------
            if save_LEH:
                img_L = util.single2uint(img_L)
                k_v = k / np.max(k) * 1.0
                k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
                k_v = cv2.resize(k_v, (3 * k_v.shape[1], 3 * k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_I = cv2.resize(img_L, (sf * img_L.shape[1], sf * img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_I[:k_v.shape[0], -k_v.shape[1]:, :] = k_v
                img_I[:img_L.shape[0], :img_L.shape[1], :] = img_L
                util.imshow(np.concatenate([img_I, img_E, img_H], axis=1),
                            title='LR / Recovered / Ground-truth') if show_img else None
                util.imsave(np.concatenate([img_I, img_E, img_H], axis=1),
                            os.path.join(E_path, f"{img_name}_k{k_index}_LEH.png"))

            psnr = util.calculate_psnr(img_E, img_H, border=border)
            psnr_L = util.calculate_psnr(img_I, img_H, border=border)
            test_results['psnr'].append(psnr)
            logger.info('{:>10s}--k:{:>2d} PSNR: {:.2f}dB, PSNR_L: {:.2f}dB'.format(img_name, k_index, psnr, psnr_L))

        # --------------------------------
        # Average PSNR
        # --------------------------------
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        logger.info(
            '------> Average PSNR of ({}) for kernel: ({}) and sigma: ({:.2f})={:.2f} dB'
                .format(args.dataset, k_index, noise_level_model, ave_psnr))
        test_results_ave['psnr'].append(ave_psnr)


if __name__ == '__main__':
    main()
