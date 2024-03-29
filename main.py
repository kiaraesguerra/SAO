import argparse
from train_utils.dataloaders import get_dataloader
from train_utils.callbacks import get_callback
from train_utils.loggers import get_logger
from train_utils.models import get_model
from train_utils.utils import remove_parameters, measure_sparsity
from train_utils.train_utils import get_plmodule
from pruners.pruners import get_pruner
from ortho_initializers.initializations import get_initializer
from pytorch_lightning import Trainer
import torch

parser = argparse.ArgumentParser(
    description="Experiments for Sparsity-Aware Orthogonal Initialization (SAO)"
)


# Model Initialization
parser.add_argument("--model", type=str, default="cnn")
parser.add_argument("--num-layers", type=int, default=16)
parser.add_argument("--hidden-width", type=int, default=16)
parser.add_argument("--activation", type=str, default="tanh")
parser.add_argument("--weight-init", type=str, default="kaiming")
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--gain", type=float, default=1.0)


# Pruning
parser.add_argument("--pruning-method", type=str, default=None)
parser.add_argument("--degree", type=int, default=None)
parser.add_argument("--sparsity", type=float, default=None)


# Dataloader and augmentations
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", type=int, default=100)
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--flip", action="store_true")
parser.add_argument("--num_workers", default=4, type=int)


# Training
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--warmup-epochs", type=int, default=0)


# Criterion
parser.add_argument("--criterion", type=str, default="crossentropy")
parser.add_argument("--label-smoothing", type=float, default=0)


# Optimizer settings
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--nesterov", action="store_true")
parser.add_argument("--amsgrad", action="store_true")
parser.add_argument("--eps", type=float, default=1.0e-08)
parser.add_argument("--gamma", default=0.1, type=float)


# Adam hyperparameters
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.99)


# Regularization
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--drop-out", type=float, default=1e-4)


# Learning rate and scheduler
parser.add_argument("--max-lr", default=0.1, type=float)
parser.add_argument("--min-lr", default=1e-6, type=float)
parser.add_argument("--scheduler", type=str, default="multistep")
parser.add_argument("--milestones", default=[100, 150], type=list)


# Callbacks
parser.add_argument("--callbacks", type=list, default=["checkpoint"])
parser.add_argument("--ckpt_path", type=str, default=None)
parser.add_argument("--ckpt_path_resume", type=str, default=None)
parser.add_argument("--save-top-k", type=int, default=1)
parser.add_argument("--save-last", action="store_false")
parser.add_argument("--experiment-name", type=str, default="experiment")
parser.add_argument("--filename", type=str, default="best")
parser.add_argument("--dirpath", type=str, default="results")

args = parser.parse_args()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    train_dl, validate_dl, test_dl = get_dataloader(args)
    model = get_model(args)
    model = get_initializer(model, args)

    if args.sparsity or args.degree:
        model = get_pruner(model, args)

    model = get_plmodule(model, args)
    callbacks = get_callback(args)
    logger = get_logger(args)
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(model, train_dl, validate_dl, ckpt_path=args.ckpt_path)
    trainer.test(dataloaders=test_dl, ckpt_path=args.ckpt_path)
    print(f"Model sparsity = {measure_sparsity(model)}")
    ckpt_path = callbacks[0].best_model_path
    model_checkpoint = torch.load(ckpt_path)
    model.load_state_dict(model_checkpoint["state_dict"])
    if args.pruning_method:
        remove_parameters(model)
    torch.save(
        model.state_dict(),
        f"{args.dirpath}/{args.experiment_name}/{args.experiment_name}.pt",
    )
