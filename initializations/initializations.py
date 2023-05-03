from .eco import *
from .kaiming import *


def get_initializer(model, args):
    if args.weight_init == "eco":
        print("ECO init")
        model = ECO_Init(
            model,
            gain=args.gain,
            sparsity=args.sparsity,
            degree=args.degree,
            activation=args.activation,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
        )
    elif args.weight_init == "delta-eco":
        print("Delta-ECO init")
        model = Delta_ECO_Init(
            model,
            gain=args.gain,
            sparsity=args.sparsity,
            degree=args.degree,
            activation=args.activation,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
        )

    elif args.weight_init == "kaiming-normal":
        print("Kaiming init")
        model = Kaiming_Init_Func(model, args)

    return model
