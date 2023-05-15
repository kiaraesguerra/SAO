from .init_calls import *

def get_initializer(model, args):
    if args.weight_init == "eco":
        print("Initializing model with ECO")
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
        print("Initializing model with Delta-ECO")
        model = Delta_ECO_Init(
            model,
            gain=args.gain,
            sparsity=args.sparsity,
            degree=args.degree,
            activation=args.activation,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
        )

    elif args.weight_init == 'delta':
        print("Initializing model with Delta")
        model = Delta_Init(
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
