from initializations.eco import ECO_Init
from pruners.standard_pruning import *


ramanujan_ = [
    "SAO",
    "SAO-relu",
    "RG",
    "RG-U-relu",
    "RG-N-relu",
]
standard_ = ["LMP", "LRP"]


def Standard_Pruning_Func(model, **kwargs):
    pruningMethod = Standard_Pruning(model, **kwargs)
    model = pruningMethod()
    return model


def get_pruner(model, args):
    if args.pruning_method in ramanujan_:
        model = ECO_Init(
            model,
            gain=args.gain,
            method=args.pruning_method,
            sparsity=args.sparsity,
            degree=args.degree,
            activation=args.activation,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
        )
        

    elif args.pruning_method in standard_:
        model = Standard_Pruning_Func(
            model,
            pruner=args.pruning_method,
            sparsity=args.sparsity,
            degree=args.degree,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
        )

    return model
