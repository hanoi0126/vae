import torch
import torch.nn.functional as F

def criterion(predict, target, ave, log_dev):
    bce_loss = F.binary_cross_entropy(predict, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
    loss = bce_loss + kl_loss
    return loss

def criterion_ae(predict, target):
    loss = F.binary_cross_entropy(predict, target, reduction='sum')
    return loss