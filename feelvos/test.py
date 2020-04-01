import random

import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from feelvos.models.Backbone import UNet
from feelvos.dataset import FEELVOSTriple
from feelvos.transform import preprocessing


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == "__main__":
    target_folder = './data/'
    ds_test = FEELVOSTriple(root='./data/', split='test', transform=preprocessing)

    loc = './unet/weight010'
    model = UNet(3, 1)
    model.load_state_dict(torch.load(loc+'.pt'))
    model = model.to(device)
    model.eval()

    pick = []
    for i in range(1):
        pick.append(random.randrange(0, 500, 1))

    for i in pick:
        X, y = ds_test.__getitem__(i)
        torchvision.utils.save_image(X[0], './testimage/'+str(i)+'_X'+'.png')
        torchvision.utils.save_image(y[0], './testimage/'+str(i)+'_y'+'.png')
        X = X[0].view(1, 3, 256, 256).cuda()
        y_pred = model(X)
        torchvision.utils.save_image(y_pred, './testimage/'+loc.split('/')[-1]+'_'+str(i)+'_ypred'+'.png')
