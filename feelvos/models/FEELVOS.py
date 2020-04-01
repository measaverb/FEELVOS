from cv2 import cv2
import torch
import torch.nn as nn
import torchvision
from modelsummary import summary
from feelvos.models.Backbone import UNet
from feelvos.models.Embeddings import PixelwiseEmbedding
from feelvos.models.DynamicSegmentationHead import DynamicSegmentationHead
from feelvos.models.Matching import global_matching, local_matching


class FEELVOS(nn.Module):
    def __init__(self, c_in, n_classes, use_gt=True, pretrained=None):
        super(FEELVOS, self).__init__()
        self.n_classes = n_classes
        self.use_gt = use_gt
        self.backbone = UNet(c_in, n_classes)
        if pretrained is not None:
            self.backbone.load_state_dict(torch.load(pretrained))
            self.backbone.eval()
        self.embedding = PixelwiseEmbedding(n_classes, n_classes, 100)
        self.dsh = DynamicSegmentationHead(n_classes+1+1+1, 1)

    def forward(self, x_list):
        x1 = x_list[0]
        x2 = x_list[1]
        x3 = x_list[2]

        if self.use_gt == False:
            x1 = self.backbone(x1)
            x2 = self.backbone(x2)
        x3 = self.backbone(x3)

        x1_l = []; x1_e = []
        x2_l = []; x2_e = []
        x3_l = []; x3_e = []
        gm = []; lm = []
        logits = []

        for i in range(self.n_classes):
            x1_l.append(x1[:, i, :, :].unsqueeze(1))
            x1_e.append(self.embedding(x1_l[i]))
            x2_l.append(x2[:, i, :, :].unsqueeze(1))
            x2_e.append(self.embedding(x2_l[i]))
            x3_l.append(x3[:, i, :, :].unsqueeze(1))
            x3_e.append(self.embedding(x3_l[i]))
            gm.append(global_matching(x1_e[i], x3_e[i]))
            lm.append(global_matching(x2_e[i], x3_e[i]))
            x_t = torch.cat((x3, gm[i].cuda(), lm[i].cuda(), x2_l[i]), dim=1)
            logits.append(self.dsh(x_t))
        x = None
        for i in range(self.n_classes):
            if i == 0:
                x = logits[i]
            else:
                x = torch.cat((logits[i-1], logits[i]), dim=1)
        return x


if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = FEELVOS(3, 1, use_gt=False).cuda(device=device)

    # summary(model, torch.zeros((1, 3, 512, 512)).cuda(), show_input=True)
    # summary(model, torch.zeros((1, 3, 512, 512)).cuda(), show_input=False)

    x1 = cv2.imread('example/x2.png')
    x2 = cv2.imread('example/x3.png')

    x1 = cv2.resize(x1, dsize=(256, 256))
    x1 = torchvision.transforms.ToTensor()(x1)
    x1 = x1.unsqueeze(0).to(device=device)

    x2 = cv2.resize(x2, dsize=(256, 256))
    x2 = torchvision.transforms.ToTensor()(x2)
    x2 = x2.unsqueeze(0).to(device=device)

    x = torch.cat((x1, x2), dim=0)
    y = model(x, x, x)
    print(y)
