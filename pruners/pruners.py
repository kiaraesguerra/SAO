from pruners.standard_pruning import *
from initializations.init_calls import *


ramanujan_ = [
    "SAO",
    "RG",
    "RG-U-relu",
    "RG-N-relu",
]

standard_ = ["LMP", "LRP", "lmp", "lrp"]


def Standard_Pruning_Func(model, **kwargs):
    pruningMethod = Standard_Pruning(model, **kwargs)
    model = pruningMethod()
    return model


def get_pruner(model, args):
    if args.pruning_method in ramanujan_ and "lip" in args.model:
        model = Delta_ECO_Init(
            model,
            gain=args.gain,
            method=args.pruning_method,
            sparsity=args.sparsity,
            degree=args.degree,
            activation=args.activation,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
        )

    elif args.pruning_method in ramanujan_:
        model = Delta_Init(
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
