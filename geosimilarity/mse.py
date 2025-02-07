import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    
    def forward(self, Xc, Xt):
        loss = torch.mean((Xc - Xt) ** 2)
        return loss