import os
import cv2

import torch
import random
import numpy as np
from torch.utils.data import Dataset
import utils.utils_image as util


class testDataset(Dataset):
    def __init__(self, root, sigma=25):
        self.root = root
        self.paths = []
        self.names = []
        self.sigma_test = sigma

        for root, dirs, fnames in os.walk(self.root):
            for fname in fnames:
                img_path = os.path.join(root, fname)
                self.names.append(fname)
                self.paths.append(img_path)
        self.len = len(self.paths)

    def __getitem__(self, item):
        img = cv2.imread(self.paths[item], cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # numpy uint
        h, w = img.shape[:2]
        img_H = util.uint2single(img) # numpy float
        img_L = np.copy(img_H)
        np.random.seed(seed=0)

        noise_level = torch.FloatTensor([self.sigma_test]) / 255.0
        img_L += np.random.normal(0, noise_level, img_L.shape)
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        noise_map = noise_level.repeat(1, h, w)
        return {'L': img_L, 'H': img_H, 'noise': noise_map, 'name':self.names[item]}

    def __len__(self):
        return self.len


class trainDataset(Dataset):
    def __init__(self, root, patch_size=128, sigma=(0, 50)):
        self.root = root
        self.paths = []
        self.names = []
        self.sigma = sigma
        self.sigma_min, self.sigma_max = self.sigma[0], self.sigma[1]
        self.patch_size = patch_size

        for root, dirs, fnames in os.walk(self.root):
            for fname in fnames:
                img_path = os.path.join(root, fname)
                self.names.append(fname)
                self.paths.append(img_path)

        self.len = len(self.paths)

    def __getitem__(self, item):
        img = cv2.imread(self.paths[item], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # numpy uint

        # random crop patch
        H, W = img.shape[:2]
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        patch_H = img[rnd_h:rnd_h + self.patch_size,
                    rnd_w:rnd_w + self.patch_size,
                    :]

        # augment the img patch(uint)
        mode = np.random.randint(0, 8)
        patch_H = util.augment_img(patch_H, mode=mode)

        # HWC to CHW, numpy(uint) to tensor
        patch_H = util.uint2tensor3(patch_H)
        patch_L = patch_H.clone()

        noise_level = torch.FloatTensor([np.random.uniform(self.sigma_min, self.sigma_max)]) / 255.0
        noise = torch.randn(patch_L.size()).mul_(noise_level).float()

        noise_map = noise_level.repeat(1,self.patch_size,self.patch_size)
        patch_L.add_(noise)

        return {'L': patch_L, 'H': patch_H, 'noise': noise_map, 
        'name':self.names[item], 'level':noise_level*255}

    def __len__(self):
        return self.len


if __name__ == '__main__':
    testset = testDataset(root='../testsets')
    print(len(testset))
    print(testset[43]['L'].shape)
