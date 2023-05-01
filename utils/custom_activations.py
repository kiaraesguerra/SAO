import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxMin(nn.Module):
    def __init__(self):
        super(MaxMin, self).__init__()

    def forward(self, z, axis=1):
        a, b = z.split(z.shape[axis] // 2, axis)
        c, d = torch.max(a, b), torch.min(a, b)
        return torch.cat([c, d], dim=axis)
    