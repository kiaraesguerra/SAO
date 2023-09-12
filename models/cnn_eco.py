import torch.nn as nn
import torch.nn.functional as F
import einops
from models.activations import activation_dict

__all__ = ["cnn_eco"]


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


class CNN_ECO(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels_0,
        activation,
        num_layers,
        hidden_width,
        num_classes=10,
    ):
        super(CNN_ECO, self).__init__()

        self.image_size = image_size
        self.in_channels_0 = in_channels_0
        self.activation = activation_dict[activation]
        self.num_layers = num_layers
        self.hidden_width = hidden_width
        self.num_classes = num_classes
        self.feature_extractor = self.make_layers()
        self.fc = nn.Linear(hidden_width, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def make_layers(self):
        layers = []
        in_channels = 3
        for stride in [1, 2, 2]:
            conv2d = nn.Conv2d(
                in_channels, self.hidden_width, kernel_size=3, padding=1, stride=stride
            )
            layers += [conv2d, self.activation]
            in_channels = self.hidden_width

        for i in range(self.num_layers):
            if i == self.num_layers // 2 - 1:
                conv2d = ECO(
                    in_channels=self.hidden_width,
                    out_channels=self.hidden_width,
                    kernel_size=3,
                    stride=3,
                    padding=3,
                    dilation=3,
                    padding_mode="circular",
                )
            elif i > self.num_layers - 2:
                conv2d = ECO(
                    in_channels=self.hidden_width,
                    out_channels=self.hidden_width,
                    kernel_size=3,
                    stride=3,
                    padding=1,
                    dilation=1,
                    padding_mode="circular",
                )

            else:
                if i < 2:
                    conv2d = nn.Conv2d(
                        in_channels,
                        self.hidden_width,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                    )
                elif i > self.num_layers // 2 - 1:
                    conv2d = nn.Conv2d(
                        self.hidden_width,
                        self.hidden_width,
                        kernel_size=3,
                        padding=1,
                        dilation=1,
                        padding_mode="circular",
                    )
                else:
                    conv2d = nn.Conv2d(
                        self.hidden_width,
                        self.hidden_width,
                        kernel_size=3,
                        padding=3,
                        dilation=3,
                        padding_mode="circular",
                    )

            layers += [conv2d, self.activation]

        return nn.Sequential(*layers)


def cnn_eco(**kwargs):
    """Constructs a 8 layers vanilla model."""
    model = CNN_ECO(**kwargs)
    return model
