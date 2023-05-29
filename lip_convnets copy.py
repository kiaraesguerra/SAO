import torch.nn.functional as F
import torch.nn as nn
import torch
import einops


class ECO(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, 
                 bias=True):
        super(ECO, self).__init__()
        assert (stride==1) or (stride==2) or (stride==3)

        self.out_channels = out_channels
        self.in_channels = in_channels*stride*stride
        self.max_channels = max(self.out_channels, self.in_channels)
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(self.max_channels, self.max_channels, self.kernel_size, 
                                               1)

    def forward(self, x):

        if self.stride > 1:
            x = einops.rearrange(x, "b c (w k1) (h k2) -> b (c k1 k2) w h", 
                                 k1=self.stride, k2=self.stride)
        if self.out_channels > self.in_channels:
            diff_channels = self.out_channels - self.in_channels
            p4d = (0, 0, 0, 0, 0, diff_channels, 0, 0)
            curr_z = F.pad(x, p4d)
        else:
            curr_z = x
 

        self.conv.dilation = max(1, x.shape[-1] // self.kernel_size)
        if self.kernel_size > 1:
            self.conv.padding = max(1, x.shape[-1] // self.kernel_size)
        
        print(self.conv.dilation)
        print(self.conv.padding)
        curr_z = self.conv(curr_z)
        z = curr_z
        if self.out_channels < self.in_channels:
            z = z[:, :self.out_channels, :, :]
            
        return z


class NormalizedLinear(nn.Linear):
    def forward(self, X):
        X = X.view(X.shape[0], -1)
        weight_norm = torch.norm(self.weight, dim=1, keepdim=True)
        self.lln_weight = self.weight/weight_norm
        return F.linear(X, self.lln_weight if self.training else self.lln_weight.detach(), self.bias)

class LipBlock(nn.Module):
    def __init__(self, in_planes, planes, conv_layer, activation_name, stride=1, kernel_size=3):
        super(LipBlock, self).__init__()
        self.conv = conv_layer(in_planes, planes, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size//2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.conv(x))
        return x
        
class LipConvNet(nn.Module):
    def __init__(self, activation, init_channels=32, block_size=1, 
                 num_classes=10, input_side=32, lln=False):
        super(LipConvNet, self).__init__()        
        self.lln = lln
        self.in_planes = 3
        
        assert type(block_size) == int

        self.layer1 = self._make_layer(init_channels, block_size, nn.Conv2d, 
                                       activation, stride=2, kernel_size=3)
        self.layer2 = self._make_layer(self.in_planes, block_size, nn.Conv2d, 
                                       activation, stride=2, kernel_size=3)
        self.layer3 = self._make_layer(self.in_planes, block_size, ECO, 
                                       activation, stride=2, kernel_size=3)
        self.layer4 = self._make_layer(self.in_planes, block_size, ECO,
                                       activation, stride=3, kernel_size=3)
        self.layer5 = self._make_layer(self.in_planes, block_size, ECO, 
                                       activation, stride=1, kernel_size=1)
        
        flat_size = input_side // 32
        flat_features = flat_size * flat_size * self.in_planes
        if self.lln:
            self.last_layer = NormalizedLinear(flat_features, num_classes)

    def _make_layer(self, planes, num_blocks, conv_layer, activation, 
                    stride, kernel_size):
        strides = [1]*(num_blocks-1) + [stride]
        kernel_sizes = [3]*(num_blocks-1) + [kernel_size]
        layers = []
        for stride, kernel_size in zip(strides, kernel_sizes):
            layers.append(LipBlock(self.in_planes, planes, conv_layer, activation, 
                                   stride, kernel_size))
            self.in_planes = planes #* stride
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.last_layer(x)
        x = x.view(x.shape[0], -1)
        return x
    