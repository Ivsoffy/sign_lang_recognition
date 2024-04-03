# import torch.nn.functional as F
import torch.nn.functional as F
import torch.nn as nn
import torch

# def nll_loss(output, target):
#     return F.nll_loss(output, target)
def focal_loss(output, target):
    loss = nn.CrossEntropyLoss()
    alpha = 1
    gamma = 2
    CE_loss = loss(output, target)
    pt = torch.exp(-CE_loss) 
    F_loss = alpha * (1-pt)**gamma * CE_loss
    return F_loss
