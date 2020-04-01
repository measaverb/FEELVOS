import torch
import torch.nn as nn
from feelvos.models.Embeddings import DepthwiseSeparableConv2D


class DynamicSegmentationHead(nn.Module):
    def __init__(self, cin, cout):
        super(DynamicSegmentationHead, self).__init__()
        self.depthwise_l = DepthwiseSeparableConv2D(cin, 256, 7)
        self.depthwise_r = DepthwiseSeparableConv2D(256, 256, 7)
        self.conv = nn.Conv2d(256, cout, 1)

    def forward(self, x):
        x = self.depthwise_l(x)
        x = self.depthwise_r(x)
        x = self.depthwise_r(x)
        x = self.depthwise_r(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv(x)
        x = nn.Softmax2d()(x)

        return x