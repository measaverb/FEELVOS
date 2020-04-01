from itertools import combinations
from cv2 import cv2
import os
import natsort
import pandas as pd
import numpy as np

import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from torchvision import transforms, utils

from feelvos.transform import preprocessing


class FEELVOSTriple(Dataset):
    def __init__(self, root='./data/', split='train', transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.folder_list = []
        self.items = []

        folder_f = open(os.path.join(root, self.split+"_folder_list.txt"), "r")
        for x in folder_f:
            self.folder_list.append(x[:-1])

        for i in range(len(self.folder_list)):
            tmp_list = natsort.natsorted(os.listdir(os.path.join(root, 'image', self.folder_list[i])))
            for j in range(len(tmp_list) - 2):
                first = tmp_list[j]
                for k in range(len(tmp_list[j+1:])-1):
                    comb_1 = tmp_list[k+1]
                    comb_2 = tmp_list[k+2]
                    self.items.append((os.path.join(self.root, 'image', self.folder_list[i], first), os.path.join(self.root, 'image', self.folder_list[i], comb_1), os.path.join(self.root, 'image', self.folder_list[i], comb_2)))

    def __getitem__(self, index):
        src = []
        mask = []
        seltem = self.items[index]
        for i in range(3):
            src.append(cv2.imread(seltem[i]))
            mask.append(cv2.imread(os.path.join(seltem[i].split('/')[1], 'mask', seltem[i].split('/')[3], seltem[i].split('/')[4])))
        sample = (src, mask)
        if self.transform is None:
            pass
        else:
            sample = self.transform(*sample)
        if self.split == 'train':
            sample[0][0] = sample[1][0]
            sample[0][1] = sample[1][1]
        return sample

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    ds_train = FEELVOSTriple(root='./data/', split='train', transform=preprocessing)
    ds_test = FEELVOSTriple(root='./data/', split='test', transform=preprocessing)
    print("DATA LOADED")
