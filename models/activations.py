import torch.nn as nn

activation_dict = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax(dim=1),
    "softplus": nn.Softplus(),
    "softsign": nn.Softsign(),
    "hardshrink": nn.Hardshrink(),
    "leakyrelu": nn.LeakyReLU(),
    "logsigmoid": nn.LogSigmoid(),
    "prelu": nn.PReLU(),
    "relu6": nn.ReLU6(),
    "rrelu": nn.RReLU(),
    "selu": nn.SELU(),
    "celu": nn.CELU(),
    "gelu": nn.GELU(),
    "hardtanh": nn.Hardtanh(),
    "tanhshrink": nn.Tanhshrink(),
    "softshrink": nn.Softshrink(),
}
