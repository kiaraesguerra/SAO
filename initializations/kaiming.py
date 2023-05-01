import torch.nn as nn
import torch


def Kaiming_Init_Func(model, args):
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity=args.activation
            )
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 1)

    return model
