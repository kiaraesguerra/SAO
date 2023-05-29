import torch.nn as nn


def get_criterion(args):
    if args.criterion in ["CrossEntropy", "crossentropy"]:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    elif args.criterion in ["BCE", "bce"]:
        criterion = nn.BCELoss()

    elif args.criterion in ["L1", "l1"]:
        criterion = nn.L1Loss()

    return criterion
