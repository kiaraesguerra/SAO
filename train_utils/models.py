import models

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def get_model(args):
    if "resnet" in args.model:
        model = models.__dict__[args.model](num_classes=args.num_classes)
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.model))
            model = models.__dict__[args.model](
                num_classes=args.num_classes, pretrained=True
            )

    elif "mlp" in args.model or "cnn" in args.model:
        model = models.__dict__[args.model](
            hidden_width=args.hidden_width,
            num_classes=args.num_classes,
            activation=args.activation,
            image_size=args.image_size,
            num_layers=args.num_layers,
            in_channels_0=args.in_channels_0,
        )
    elif "trial" in args.model:
        model = models.__dict__[args.model]()

    return model.to(args.device)
