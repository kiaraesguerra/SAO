import torch.nn as nn
from models.activations import activation_dict


class MLP(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        in_channels_0: int = 3,
        num_classes: int = 10,
        activation: str = "relu",
        num_layers: int = 5,
        hidden_width: int = 128,
    ):
        super(MLP, self).__init__()

        self.image_size = image_size
        self.num_classes = num_classes
        self.in_channels_0 = in_channels_0
        self.activation = activation_dict[activation]
        self.num_layers = num_layers
        self.hidden_width = hidden_width
        self.input_layer = nn.Linear(
            image_size * image_size * in_channels_0, hidden_width
        )
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_width, hidden_width) for _ in range(num_layers - 1)]
        )
        self.output_layer = nn.Linear(hidden_width, num_classes)

    def forward(self, x):
        x = x.view(-1, self.image_size * self.image_size * self.in_channels_0)
        x = self.input_layer(x)
        x = self.activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.output_layer(x)

        return x


def mlp(**kwargs):
    return MLP(**kwargs)
