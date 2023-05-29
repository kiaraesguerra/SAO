import torch.nn as nn
import torch.nn.functional as F
import einops

__all__ = ["vanlip8", "vanlip12", "vanlip16"]


class ECO(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=None,
        bias=True,
        padding_mode="circular",
    ):
        super(ECO, self).__init__()
        assert (stride == 1) or (stride == 2) or (stride == 3)

        self.out_channels = out_channels
        self.in_channels = in_channels * stride * stride
        self.max_channels = max(self.out_channels, self.in_channels)
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            self.max_channels,
            self.max_channels,
            self.kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            padding_mode=padding_mode,
            bias=bias,
        )

    def forward(self, x):
        if self.stride > 1:
            x = einops.rearrange(
                x,
                "b c (w k1) (h k2) -> b (c k1 k2) w h",
                k1=self.stride,
                k2=self.stride,
            )
        if self.out_channels > self.in_channels:
            diff_channels = self.out_channels - self.in_channels
            p4d = (0, 0, 0, 0, 0, diff_channels, 0, 0)
            curr_z = F.pad(x, p4d)
        else:
            curr_z = x

        curr_z = self.conv(curr_z)
        z = curr_z
        if self.out_channels < self.in_channels:
            z = z[:, : self.out_channels, :, :]

        return z


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
            conv2d = ECO(
                in_channels=c,
                out_channels=c,
                kernel_size=3,
                stride=3,
                padding=3,
                dilation=3,
                padding_mode="circular",
            )
        elif i > depth - 2:
            conv2d = ECO(
                in_channels=c,
                out_channels=c,
                kernel_size=3,
                stride=3,
                padding=1,
                dilation=1,
                padding_mode="circular",
            )

        else:
            if i < 2:
                conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1, stride=1)

            elif i > depth // 2 - 1:
                conv2d = nn.Conv2d(
                    c, c, kernel_size=3, padding=1, dilation=1, padding_mode="circular"
                )
            else:
                conv2d = nn.Conv2d(
                    c, c, kernel_size=3, padding=3, dilation=3, padding_mode="circular"
                )

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
