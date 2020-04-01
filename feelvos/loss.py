import torch

def dice_loss(pred, target, epsilon=1e-7, use_sigmoid=True):
    pred = pred.contiguous()
    if use_sigmoid:
        pred = torch.nn.Sigmoid()(pred)
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + epsilon) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + epsilon)))
    return loss.mean()
    
def tversky_loss(pred, target, epsilon=1e-7):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    false_neg = (target * (1 - pred)).sum(dim=2).sum(dim=2)
    false_pos = ((1 - target) * pred).sum(dim=2).sum(dim=2)
    alpha = 0.7
    loss = (1 - ((intersection + epsilon)/(intersection + alpha * false_neg + (1-alpha)*false_pos + epsilon)))
    return loss.mean()