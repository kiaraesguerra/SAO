from pruners.standard_pruning import *
from ortho_initializers.init_calls import *

ramanujan_ = [
    "sao",
    "rg",
    "rg-u",
    "rg-n",
]

standard_ = ["lmp", "lrp"]


def Standard_Pruning_Func(model, **kwargs):
    pruningMethod = Standard_Pruning(model, **kwargs)
    model = pruningMethod()
    return model


def get_pruner(model, args):
    if args.pruning_method.lower() in ramanujan_ and "lip" in args.model:
        model = Delta_ECO_Init(
            model,
            gain=args.gain,
            method=args.pruning_method,
            sparsity=args.sparsity,
            degree=args.degree,
            activation=args.activation,
            in_channels_0=args.in_channels_0,
            num_classes=args.num_classes,
        )

    elif args.pruning_method.lower() in ramanujan_:
        model = Delta_Init(
            model,
            gain=args.gain,
            method=args.pruning_method,
            sparsity=args.sparsity,
            degree=args.degree,
            activation=args.activation,
            in_channels_0=args.in_channels_0,
            num_classes=args.num_classes,
        )

    elif args.pruning_method.lower() in standard_:
        model = Standard_Pruning_Func(
            model,
            pruner=args.pruning_method,
            sparsity=args.sparsity,
            degree=args.degree,
            in_channels_0=args.in_channels_0,
            num_classes=args.num_classes,
        )

    return model
