import torch.nn as nn
from models.activations import activation_dict


class CNN(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels_0,
        activation,
        num_layers,
        hidden_width,
        num_classes=10,
    ):
        super(CNN, self).__init__()

        self.image_size = image_size
        self.in_channels_0 = in_channels_0
        self.activation = activation_dict[activation]
        self.num_layers = num_layers
        self.hidden_width = hidden_width
        self.num_classes = num_classes
        self.feature_extractor = self.make_layers()
        self.fc = nn.Linear(hidden_width * 4, num_classes)

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
            stride = (
                2 if (i == self.num_layers // 2 - 1) or (i > self.num_layers - 2) else 1
            )
            conv2d = nn.Conv2d(
                self.hidden_width,
                self.hidden_width,
                kernel_size=3,
                padding=1,
                stride=stride,
            )
            layers += [conv2d, self.activation]

        return nn.Sequential(*layers)


def cnn(**kwargs):
    """Constructs a plain CNN."""
    model = CNN(**kwargs)
    return model
