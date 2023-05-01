from .eco import *
from .kaiming import *


def get_initializer(model, args):
    if "eco" in args.weight_init:
        model = ECO_Init(model,
                         sparsity=args.sparsity,
                         degree=args.degree,
                         activation=args.activation,
                         in_channels=args.in_channels,
                         num_classes=args.num_classes)
    elif args.weight_init == "kaiming-normal":
        model = Kaiming_Init_Func(model, args)

    return model
