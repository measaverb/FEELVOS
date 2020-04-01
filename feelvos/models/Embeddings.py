import torch
import torch.nn as nn
import torch.nn.functional as F 
from modelsummary import summary


class DepthwiseSeparableConv2D(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableConv2D,self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_in, kernel_size, stride, padding, dilation, groups=c_in, bias=bias)
        self.pointwise = nn.Conv2d(c_in, c_out, 1, 1, 0, 1, 1, bias=bias)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class PixelwiseEmbedding(nn.Module):
    def __init__(self, c_in, c_out_1, c_out_2):
        super(PixelwiseEmbedding, self).__init__()
        self.separable = DepthwiseSeparableConv2D(c_in=c_in, c_out=c_out_1, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(c_out_1, c_out_2, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.separable(x)
        x = self.conv1(x)
        return x
