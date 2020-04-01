import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_ch, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_ch = n_ch
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = Double33Conv(n_ch, 64)
        
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out = Out(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        y = self.out(x)
        return y
    

class Double33Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double33conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double33conv(x)
    

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            Double33Conv(in_ch, out_ch)
        )
    
    def forward(self, x):
        return self.down(x)
    

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:            
            self.up = nn.ConvTranspose2d(in_ch//2, out_ch//2, kernel_size=2, stride=2)
        
        self.conv = Double33Conv(in_ch, out_ch)
                                         
    
    def forward(self, x1, x2):
        x1 = self.up(x1)

        dy = x2.size()[2]-x1.size()[2]
        dx = x2.size()[3]-x1.size()[3]
        """ Caution: Padding dimension
        N, C, H, W, dx=diffence of W-value
        pad=(w_left,w_right,h_top,h_bottom)
        """
        x1 = F.pad(input=x1, pad=(dx//2, dx-dx//2, dy//2, dy-dy//2))
        # print('sizes',x1.size(),x2.size(),dx // 2, dx - dx//2, dy // 2, dy - dy//2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class Out(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.out = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    
    def forward(self, x):
        return self.out(x)