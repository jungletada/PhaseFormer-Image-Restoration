import cv2
import os
import argparse
import numpy as np
from utils import utils_image as util

"""
Code from free domain website,
for enlarging image results.
Modified by Dingjie Peng
"""

i, j, h, w = 177, 689, 128, 128
eh, ew = 240, 240
img_name = 'img_012_k0_LEH.png'


def args_pharse():
    parser = argparse.ArgumentParser(description="Test for image deblurring")
    parser.add_argument('--ckpt', type=int, default=1100, help='load checkpoint')
    parser.add_argument('--dataset', type=str, default='Urban100', help='Gaussian noise level')
    parser.add_argument('--name', type=str, default='phaseformer_S', help='model name')
    parser.add_argument('--downsample', type=int, default=4, help='downsample rate of 2^n')
    parser.add_argument('--task', type=str, default='deblur', help='choose task')
    args = parser.parse_args()
    return args


def split_imgs(img):
    _, w, _ = img.shape
    L = img[:, :w//3, :]
    E = img[:, w//3:int(2*w/3), :]
    H = img[:, int(2*w/3):, :]
    return L, E, H


def combine_imgs(imgs):
    return np.concatenate(imgs, axis=1)


def enlarge_patch(img, bbcolor=(0, 0, 255)):
    H, W, _ = img.shape
    patch = img[i:i + h, j:j + w]
    patch = cv2.resize(patch, (ew, eh))
    pt1 = (j, i)
    pt2 = (j + w, i + h)
    cv2.rectangle(img, pt1, pt2, bbcolor, 2)
    img[H - eh:, :ew, :] = patch[:, :, :]
    cv2.rectangle(img, (0, H - eh), (ew, H), bbcolor, 2)
    return img


def main():
    args = args_pharse()
    image_root = "./results"
    save_root = "./paper_results"
    result_name = f"{args.dataset}_{args.task}"
    image_path = os.path.join(image_root, result_name)
    save_path = os.path.join(save_root, result_name)
    util.mkdir(save_path)

    img_load_path = os.path.join(image_path, img_name)
    img = cv2.imread(img_load_path, -1)
    L, E, H = split_imgs(img)
    L = enlarge_patch(L)
    E = enlarge_patch(E)
    H = enlarge_patch(H)
    out = combine_imgs((L, E, H))
    cv2.imwrite(os.path.join(save_path, f"{img_name}_zoom_.png"), out)


if __name__ == '__main__':
    main()
