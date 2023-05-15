import torch.nn as nn
import torch
from initializations.eco import ECO_Constructor
from initializations.delta import Delta_Constructor


def Delta_Init(model, **kwargs):
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            vals = Delta_Constructor(module, **kwargs)
            if isinstance(vals, tuple):
                module.weight = nn.Parameter(vals[0])
                torch.nn.utils.prune.custom_from_mask(module, "weight", vals[1])
            else:
                module.weight = nn.Parameter(vals)
                
        elif isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 1)

    return model


def Delta_ECO_Init(model, **kwargs):
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.padding_mode !='cyclic':
            vals = Delta_Constructor(module, **kwargs)
            if isinstance(vals, tuple):
                module.weight = nn.Parameter(vals[0])
                torch.nn.utils.prune.custom_from_mask(module, "weight", vals[1])
            else:
                module.weight = nn.Parameter(vals)
            
        elif isinstance(module, nn.Conv2d):
            module.weight = nn.Parameter(ECO_Constructor(module, **kwargs))
            mask = (module.weight != 0)*1
            torch.nn.utils.prune.custom_from_mask(module, "weight", mask)

        elif isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 1)

    return model


def ECO_Init(model, **kwargs):
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight = nn.Parameter(ECO_Constructor(module, **kwargs))
            mask = (module.weight != 0)*1
            torch.nn.utils.prune.custom_from_mask(module, "weight", mask)
            
        elif isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 1)

    return model


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