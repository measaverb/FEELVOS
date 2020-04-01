from cv2 import cv2
import torch
import torch.nn as nn
import torchvision
from torch.autograd.variable import Variable
from .correlation_package.correlation import Correlation


def distance(p, q):
    ps = torch.sum(p * p)
    qs = torch.sum(q * q)
    norm = torch.norm(ps-qs, p=2, dim=-1)
    res = 1 - (2 / (1 + torch.exp(norm)))
    return res

def global_matching(x, y):
    output = torch.zeros(x.size(0), 1, x.size(2), x.size(3))
    for i in range(x.size(0)):
        for j in range(x.size(2)):
            for k in range(x.size(3)):
                output[i, :, j, k] = distance(x[i, :, j, k], y[i, :, j, k])
    return output

def local_matching(x, y, window):
    output = torch.zeros(x.size(0), 1, x.size(2), x.size(3))
    # out_corr = Correlation(pad_size=6, kernel_size=window, max_displacement=0, stride1=1, stride2=1, corr_multiply=1)(x, y)

    return output
