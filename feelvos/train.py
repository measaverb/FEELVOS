import argparse

from feelvos.dataset import FEELVOSTriple
from feelvos.transform import preprocessing
from feelvos.models.FEELVOS import FEELVOS
from feelvos.loss import dice_loss
from feelvos.metric import dice_coeff
from feelvos.trainer import Trainer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=7
)
parser.add_argument(
    '--epoch', type=int, default=40
)
parser.add_argument(
    '--lr', type=float, default=0.001
)
parser.add_argument(
    '--dataset', type=str, default='./data/'
)
parser.add_argument(
    '--workers', type=int, default=4
)

cfg = parser.parse_args()
print(cfg)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == "__main__":
    ds_train = FEELVOSTriple(root='./data/', split='train', transform=preprocessing)
    ds_test = FEELVOSTriple(root='./data/', split='test', transform=preprocessing)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)
    print("DATA LOADED")

    model = FEELVOS(3, 1, use_gt=True, pretrained='./unet/weight010.pt')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.BCELoss()
    success_metric = nn.BCELoss()
    summary = SummaryWriter()

    trainer = Trainer(model, criterion, optimizer, success_metric, device, None, False)
    fit = trainer.fit(dl_train, dl_test, num_epochs=cfg.epoch, checkpoints='./save2/'+model.__class__.__name__+'.pt')
    torch.save(model.state_dict(), './save/final_state_dict.pt')
    torch.save(model, './save/final.pt')

    loss_fn_name = "cross entropy"
    best_score = str(fit.best_score)
    print(f"Best loss score(loss function = {loss_fn_name}): {best_score}")
