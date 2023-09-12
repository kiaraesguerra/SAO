"Code acquired from https://github.com/singlasahil14/SOC/blob/main/custom_activations.py"
import torch
import torch.nn as nn

class MaxMin(nn.Module):
    def __init__(self):
        super(MaxMin, self).__init__()

    def forward(self, z, axis=1):
        a, b = z.split(z.shape[axis] // 2, axis)
        c, d = torch.max(a, b), torch.min(a, b)
        return torch.cat([c, d], dim=axis)