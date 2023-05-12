import torch.nn as nn

__all__ = ["vanlip8", "vanlip12", "vanlip16"]


class Vanilla_Lip(nn.Module):
    def __init__(self, base, c, num_classes=10):
        super(Vanilla_Lip, self).__init__()
        self.base = base
        self.fc = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def make_layers(depth, c, activation):
    assert isinstance(depth, int)

    if activation == "tanh":
        act = nn.Tanh()
    elif activation == "relu":
        act = nn.ReLU()

    layers = []
    in_channels = 3
    for stride in [1, 2, 2]:
        conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1, stride=stride)
        layers += [conv2d, act]
        in_channels = c
    for i in range(depth):

        if i == depth // 2 - 1:
            conv2d = ECO(in_channels=c, out_channels=c, kernel_size=3, stride=3, padding=3, dilation=3,
                 padding_mode='circular')
        elif i > depth - 2:
            conv2d = ECO(in_channels=c, out_channels=c, kernel_size=3, stride=3, padding=1, dilation=1,
                 padding_mode='circular')
          
        else:
            if i > depth // 2 - 1:
                conv2d = nn.Conv2d(c, c, kernel_size=3, padding=1, dilation=1, padding_mode='circular')
            else:
                conv2d = nn.Conv2d(c, c, kernel_size=3, padding=3, dilation=3, padding_mode='circular')
                
                  
        layers += [conv2d, act]

    return nn.Sequential(*layers), c


def vanlip8(c, activation, **kwargs):
    """Constructs a 8 layers vanilla model."""
    model = Vanilla_Lip(*make_layers(8, c, activation), **kwargs)
    return model


def vanlip12(c, activation, **kwargs):
    """Constructs a 12 layers vanilla model."""
    model = Vanilla_Lip(*make_layers(12, c, activation), **kwargs)
    return model


def vanlip16(c, activation, **kwargs):
    """Constructs a 16 layers vanilla model."""
    model = Vanilla_Lip(*make_layers(16, c, activation), **kwargs)
    return model