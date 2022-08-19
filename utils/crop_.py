#!/usr/bin/env python3
import numpy as np
import utils.utils_image as util


def split_fn(img):
    patchs = []
    patchs1 = split_img_width(img)
    for p in patchs1:
        patchs.append(split_img_height(p))
    return patchs


def split_img_height(img, stride=256):
    patchs = []
    h, w, c = img.shape
    numH = h//stride
    if h % stride != 0:
        numH = numH + 1

    for i in range(numH - 1):
        p = img[i*stride:(i+1)*stride, :, :]
        patchs.append(p)

    p = img[-stride:, :, :]
    patchs.append(p)

    return patchs


def split_img_width(img, stride=256):
    patchs = []
    h, w, c = img.shape
    numH, numW = h//stride, w//stride

    if h % stride != 0:
        numH = numH + 1
    if w % stride != 0:
        numW = numW + 1

    for i in range(numW - 1):
        p = img[:, i*stride:(i+1)*stride, :]
        patchs.append(p)

    p = img[:, -stride:, :]
    patchs.append(p)

    return patchs


def combine_fn(patches, original):
    h, w = original
    ph = []
    for pw in patches:
        c1 = combine_height(pw, h)
        ph.append(c1)

    c2 = combine_width(ph, w)
    return c2

def combine_width(patches, width, stride=256):
    r = width % stride
    last = patches[-1]
    last = last[:, -r:, :]
    patches.pop()
    patches.append(last)
    combine = np.concatenate(patches, axis=1)
    return combine


def combine_height(patches, height, stride=256):
    r = height % stride
    last = patches[-1]
    last = last[-r:, :, :]
    patches.pop()
    patches.append(last)
    combine = np.concatenate(patches, axis=0)
    return combine


if __name__ == '__main__':
    root = '../../../data/dpir/CBSD68/'
    save_root = '../'
    name = '253055.jpg'
    path = root + name
    img = util.imread_uint(path, 3)
    h, w, c = img.shape
    print("original shape: {} x {}".format(h, w))

    patches = split_fn(img)

    idx = 1
    for p1 in patches:
        for p2 in p1:
            util.imsave(p2, save_root+str(idx)+'_'+name)
            idx += 1

    c = combine_fn(patches,(h,w))
    util.imsave(c, save_root
     + str(idx) + 'ch' + name)
    psnr = util.calculate_psnr(c, img, border=0)
    print(psnr)