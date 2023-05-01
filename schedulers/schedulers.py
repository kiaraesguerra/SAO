import torch
import warmup_scheduler


def get_scheduler(optimizer, args):
    if args.scheduler == "multistep":
        base_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma
        )
    elif args.scheduler == "cosine":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.scheduler == "lambda":
        base_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer)

    elif args.scheduler == "plateau":
        base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    elif args.scheduler == "cyclic":
        base_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer)

    if args.warmup_epoch:
        scheduler = warmup_scheduler.GradualWarmupScheduler(
            optimizer,
            multiplier=1.0,
            total_epoch=args.warmup_epoch,
            after_scheduler=base_scheduler,
        )
    else:
        scheduler = base_scheduler

    return scheduler
