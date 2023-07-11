import torch.nn as nn

__all__ = ["mlp_ls"]

class MLP_LS(nn.Module):
    def __init__(self,
                 image_size: int = 32,
                 in_channels: int = 3,
                 num_classes: int = 10,
                 activation: str = 'relu',
                 num_layers: int = 5,
                 hidden_width: int = 128):
        super(MLP_LS, self).__init__()
        
        self.image_size = image_size
        self.num_input_channels = in_channels
        self.num_classes = num_classes
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        self.num_layers = num_layers
        self.hidden_width = hidden_width
        
        self.input_layer = nn.Linear(image_size * image_size * in_channels, hidden_width)
        self.hidden_layers_L = nn.ModuleList([nn.Linear(hidden_width, hidden_width) for _ in range(num_layers - 1)])
        self.hidden_layers_S = nn.ModuleList([nn.Linear(hidden_width, hidden_width) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_width, num_classes)
        
    def forward(self, x):
        x = x.view(-1, self.image_size * self.image_size * self.num_input_channels)
        y = self.input_layer(x)
        y = self.activation(y)
        
        for layer_A, layer_B in zip(self.hidden_layers_L, self.hidden_layers_S):
            y_A = layer_A(y)
            y_B = layer_B(y)
            y = y_A + y_B
            y = self.activation(y)
            
        y = self.output_layer(y)

        return y

def mlp_ls(**kwargs):
    return MLP_LS(**kwargs)
