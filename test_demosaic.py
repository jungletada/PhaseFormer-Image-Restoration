import os.path
import cv2
import logging
import numpy as np
import argparse
from collections import OrderedDict

import torch
from utils import utils_model
from utils import utils_mosaic
from utils import utils_logger
from utils import utils_pnp as pnp
from utils import utils_image as util
from model.phaseformer import build_model

"""
Modified from Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/DPIR
        https://github.com/cszn/IRCNN
        https://github.com/cszn/KAIR
"""

save_models_dir = './model_zoo'
testset_path = './data/testset'
result_path = './results'


def args_pharse():
    parser = argparse.ArgumentParser(description="Test for image deblurring")
    parser.add_argument('--ckpt', type=int, default=1100, help='load checkpoint')
    parser.add_argument('--sigma', type=float, default=0.0, help='Gaussian noise level')
    parser.add_argument('--dataset', type=str, default='McM', help='Gaussian noise level')
    parser.add_argument('--name', type=str, default='phaseformer_S', help='model name')
    parser.add_argument('--downsample', type=int, default=4, help='downsample rate of 2^n')
    parser.add_argument('--mode', type=int, default=2, help='test mode, 3: x8, 4: split x8')
    parser.add_argument('--augment', action='store_true', help='to use x8 augmentation')
    parser.add_argument('--iter', type=int, default=40, help='number of iterations')
    parser.add_argument('--show_iter_steps', action='store_true', help='To show intermediate steps')
    parser.add_argument('--save_results', action='store_true', help='To save the result images')
    args = parser.parse_args()
    return args


def main():
    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    args = args_pharse()
    noise_level_img = args.sigma     # set AWGN noise level for LR image, default: 0
    noise_level_model = noise_level_img  # set noise level of model, default: 0
    model_name = f"{args.name}_d{args.downsample}"
    x8 = args.augment        # set PGSE to boost performance, default: True
    iter_num = args.iter     # set number of iterations, default: 40 for demosaicing
    modelSigma1 = 9       # set sigma_1, default: 49
    modelSigma2 = max(0.3, noise_level_model*255.) # set sigma_2, default
    matlab_init = True

    show_img = False                     # default: False
    save_L = True                        # save LR image
    save_E = True                        # save estimated image
    save_LEH = args.save_results         # save zoomed LR, E and H images
    border = 10                          # default 10 for demosaicing
    result_name = f"{args.dataset}_demosaic"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------
    L_path = os.path.join(testset_path, args.dataset)  # L_path, for Low-quality images
    E_path = os.path.join(result_path, result_name)    # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
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

    logger.info('model_name:{}, image sigma:{:.2f}%, model sigma:{:.2f}%'
                .format(model_name, noise_level_img*100, noise_level_model*100))
    L_paths = util.get_image_paths(L_path)
    test_results = OrderedDict()
    test_results['psnr'] = []

    for idx, img in enumerate(L_paths):
        # --------------------------------
        # (1) get img_H and img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=3)
        img_H = util.modcrop(img_H, 8)  # mod crop
        CFA, CFA4, mosaic, mask = utils_mosaic.mosaic_CFA_Bayer(img_H)

        # --------------------------------
        # (2) initialize x
        # --------------------------------
        if matlab_init:  # matlab demosaicing for initialization
            CFA4 = util.uint2tensor4(CFA4).to(device)
            x = utils_mosaic.dm_matlab(CFA4)
        else:
            x = cv2.cvtColor(CFA, cv2.COLOR_BAYER_BG2RGB_EA)
            x = util.uint2tensor4(x).to(device)

        img_L = util.tensor2uint(x)
        y = util.uint2tensor4(mosaic).to(device)

        util.imshow(img_L) if show_img else None
        mask = util.single2tensor4(mask.astype(np.float32)).to(device)

        # --------------------------------
        # (3) get rhos and sigmas
        # --------------------------------
        rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.001, noise_level_img), iter_num=iter_num, lambda_=0.29,
                                         modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
        rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)
        H = util.uint2tensor4(img_H)
        # --------------------------------
        # (4) main iterations
        # --------------------------------
        for i in range(iter_num):
            # --------------------------------
            # step 1, closed-form solution
            # --------------------------------
            x = (y+rhos[i].float()*x).div(mask+rhos[i])
            # --------------------------------
            # step 2, denoiser
            # --------------------------------
            x = torch.clamp(x, 0, 1)
            if x8:
                x = util.augment_img_tensor4(x, i % 8)
                H = util.augment_img_tensor4(H, i % 8)

            N = sigmas[i].float().repeat(1, 1, x.shape[2], x.shape[3])
            x = utils_model.test_mode(model, L=x, N=N, mode=2, refield=32, min_size=128, modulo=16)
            
            if args.show_iter_steps and i % 4 == 0:
                img_inter = util.tensor2uint(x)
                img_inter_H = util.tensor2uint(H)
                util.imsave(img_inter, os.path.join(E_path, f"{img_name}_{modelname}_iter.{i}.png"))
                psnr = util.calculate_psnr(img_inter, img_inter_H, border=border)
                print("iters: {}, psnr={:.2f}".format(i, psnr))

            if x8:
                if i % 8 == 3 or i % 8 == 5:
                    x = util.augment_img_tensor4(x, 8 - i % 8)
                    H = util.augment_img_tensor4(H, 8 - i % 8)
                else:
                    x = util.augment_img_tensor4(x, i % 8)
                    H = util.augment_img_tensor4(H, i % 8)

        x[mask.to(torch.bool)] = y[mask.to(torch.bool)]
        # --------------------------------
        # (4) img_E
        # --------------------------------
        img_E = util.tensor2uint(x)
        psnr = util.calculate_psnr(img_E, img_H, border=border)
        psnr_L = util.calculate_psnr(img_L, img_H, border=border)
        logger.info('{}: PSNR_L: {:.2f} dB; PSNR_E: {:.2f} dB'.format(img_name, psnr_L, psnr))
       
        test_results['psnr'].append(psnr)
        # logger.info('{:->4d}--> {:>10s} -- PSNR: {:.2f}dB'.format(idx, img_name+ext, psnr))

        if save_E:
            util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+'.png'))

        if save_L:
            util.imsave(img_L, os.path.join(E_path, img_name+'_L.png'))

        if save_LEH:
            util.imsave(np.concatenate([img_L, img_E, img_H], axis=1), os.path.join(E_path, img_name+model_name+'_LEH.png'))

    avg_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    logger.info('------> Average PSNR(RGB) of ({}) is : {:.2f} dB'.format(args.dataset, avg_psnr))


if __name__ == '__main__':
    main()
