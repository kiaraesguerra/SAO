from initializations.eco import ECO_Init, Delta_ECO_Init
from pruners.standard_pruning import *
from sao_utils.ramanujan_constructions import Ramanujan_Construction

ramanujan_ = [
    "SAO",
    "RG",
    "RG-U-relu",
    "RG-N-relu",
]

ramanujan_delta_ = [
    "SAO-delta",
    "RG-delta",
    "RG-U-delta",
    "RG-N-delta",
]


standard_ = ["LMP", "LRP"]


def Standard_Pruning_Func(model, **kwargs):
    pruningMethod = Standard_Pruning(model, **kwargs)
    model = pruningMethod()
    return model

def Delta_Init(model, **kwargs):
    constructionMethod = Ramanujan_Construction(model, **kwargs)
    model = constructionMethod()
    return model


def get_pruner(model, args):
    if args.pruning_method in ramanujan_:
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
        
    elif args.pruning_method in ramanujan_delta_:
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
