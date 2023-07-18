import torch.nn as nn
import torch
from initializations.eco import ECO_Constructor
from initializations.delta import Delta_Constructor
from initializations.LS import LS_Constructor

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
        if isinstance(module, nn.Conv2d) and module.padding_mode != "circular":
            vals = Delta_Constructor(module, **kwargs)
            if isinstance(vals, tuple):
                module.weight = nn.Parameter(vals[0])
                torch.nn.utils.prune.custom_from_mask(
                    module, "weight", torch.abs(vals[1])
                )
            else:
                module.weight = nn.Parameter(vals)
        elif isinstance(module, nn.Conv2d) and module.padding_mode == "circular":
            module.weight = nn.Parameter(ECO_Constructor(module, **kwargs))
            torch.nn.utils.prune.custom_from_mask(module, "weight", (module.weight != 0) * 1)
        elif isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 1)

    return model


def ECO_Init(model, **kwargs):
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight = nn.Parameter(ECO_Constructor(module, **kwargs))
            torch.nn.utils.prune.custom_from_mask(module, "weight", (module.weight != 0) * 1)
        elif isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 1)

    return model



def generate_low_rank_matrix(rows, cols, rank=2):
    if rank > min(rows, cols):
        raise ValueError("Rank cannot exceed the minimum dimension.")
    U = torch.randn(rows, rank)
    V = torch.randn(rank, cols)
    matrix = torch.matmul(U, V)

    return matrix


def LS_Init(model, **kwargs):
    for _, module in model.hidden_layers_S.named_modules():
        if isinstance(module, nn.Linear):
            vals = LS_Constructor(module, **kwargs)
            if isinstance(vals, tuple):
                module.weight = nn.Parameter(vals[0])
                torch.nn.utils.prune.custom_from_mask(
                    module, "weight", torch.abs(vals[1])
                )
            else:
                module.weight = nn.Parameter(vals)
                
    for _, module in model.hidden_layers_L.named_modules():
        if isinstance(module, nn.Linear):
            low_rank_matrix = generate_low_rank_matrix(module.in_features, module.out_features)
            module.weight = nn.Parameter(low_rank_matrix)
            
            #torch.nn.init.orthogonal_(module.weight, 1)
    return model



def LS_Standard_Init(model, sparsity):
    for _, module in model.hidden_layers_S.named_modules():
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 1)
            torch.nn.utils.prune.l1_unstructured(module, name="weight", amount=sparsity)
    for _, module in model.hidden_layers_L.named_modules():
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 1)

    return model



def Kaiming_Init(model, args):
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


def low_rank(module, rank):
    u, s, v = torch.linalg.svd(module.weight)
    U_layer = nn.Linear(module.out_features, rank)
    W_layer = nn.Linear(rank, module.in_features)
    U_layer.weight = nn.Parameter(u[:, 0:rank])
    W_layer.weight = nn.Parameter(s[0:rank]@v[0:rank])
    
    revised_module = nn.Sequential(
        U_layer,
        W_layer
    )
    return revised_module


def LR_Init(model, args):
    for module_name, module in model.hidden_layers.named_modules():
        if isinstance(module, nn.Linear):
            model.hidden_layers._modules[module_name] = low_rank(module, args.rank)
    
