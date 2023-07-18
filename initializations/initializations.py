from .init_calls import *


def get_initializer(model, args):
    print(f'=> Initializing model with {args.weight_init}')
    if args.weight_init == "eco":
        model = ECO_Init(
            model,
            method=args.pruning_method,
            gain=args.gain,
            sparsity=args.sparsity,
            degree=args.degree,
            activation=args.activation,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
        )
    elif args.weight_init == "delta-eco":
        model = Delta_ECO_Init(
            model,
            method=args.pruning_method,
            gain=args.gain,
            sparsity=args.sparsity,
            degree=args.degree,
            activation=args.activation,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
        )
        
        
    elif args.weight_init == "ls":
        model = LS_Init(
            model,
            method=args.pruning_method,
            gain=args.gain,
            sparsity=args.sparsity,
            degree=args.degree,
            activation=args.activation,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
        )
        
    elif args.weight_init == "ls_standard":
        model = LS_Standard_Init(
            model,
            sparsity=args.sparsity
        )
        
    elif args.weight_init == "lr":
        model = LR_Init(
            model,
            rank=args.rank
        )

    elif args.weight_init == "delta":
        model = Delta_Init(
            model,
            method=args.pruning_method,
            gain=args.gain,
            sparsity=args.sparsity,
            degree=args.degree,
            activation=args.activation,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
        )

    elif args.weight_init == "kaiming-normal":
        model = Kaiming_Init(model, args)

    return model
