import argparse
from train_utils.dataloaders import get_dataloader
from train_utils.callbacks import get_callback
from train_utils.loggers import get_logger
from train_utils.models import get_model
from train_utils.utils import remove_parameters, measure_sparsity
from train_utils.train_utils import get_plmodule
from pruners.pruners import get_pruner
from initializations.initializations import get_initializer
from pytorch_lightning import Trainer
import models
import torch


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
parser = argparse.ArgumentParser(description="PyTorch CIFAR-10")
parser.add_argument("--model", "-a", default="van32")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", type=int, default=100, metavar="N")
parser.add_argument("--epochs", default=200, type=int, metavar="N")
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--pretrained", dest="pretrained", action="store_true")
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--min-lr", default=1e-6, type=float)
parser.add_argument("--criterion", default="crossentropy", type=str)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--nesterov", default=True, type=bool)
parser.add_argument("--milestones", default=[100, 150], type=list)
parser.add_argument("--weight-decay", default=1e-4, type=float)
parser.add_argument("--gamma", default=0.1, type=float)
parser.add_argument(
    "--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamW"]
)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.99)
parser.add_argument("--eps", type=float, default=1.0e-08)
parser.add_argument("--amsgrad", type=bool, default=False)
parser.add_argument("--image-size", type=int, default=32)
parser.add_argument("--warmup-epoch", type=int, default=0)
parser.add_argument("--scheduler", type=str, default="multistep")
parser.add_argument("--ckpt_path", type=str, default=None)
parser.add_argument("--ckpt_path_resume", type=str, default=None)
parser.add_argument("--weight-init", type=str, default="eco")
parser.add_argument("--experiment-name", type=str, default="experiment")
parser.add_argument("--autoaugment", type=bool, default=False)
parser.add_argument("--flip", type=bool, default=False)
parser.add_argument("--width", type=int, default=16)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--label-smoothing", type=float, default=0)
parser.add_argument("--save-top-k", type=int, default=1)
parser.add_argument("--save-last", type=bool, default=True)
parser.add_argument("--filename", type=str, default="best")
parser.add_argument(
    "--pruning-method",
    type=str,
    default=None
)
parser.add_argument("--degree", type=int, default=None)
parser.add_argument("--sparsity", type=float, default=None)
parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu"])
parser.add_argument("--dirpath", type=str, default="results")
parser.add_argument("--gain", type=float, default=1.0)
parser.add_argument("--callbacks", type=list, default=["checkpoint"])
parser.add_argument("--in-channels", type=int, default=3)

args = parser.parse_args()


if __name__ == "__main__":
    
    torch.backends.cudnn.benchmark=True

    train_dl, validate_dl, test_dl = get_dataloader(args)

    model = get_model(args)
    model = get_initializer(model, args)
    
    if args.sparsity or args.degree:
        model = get_pruner(model, args)
        
    print(f'Model sparsity = {measure_sparsity(model)}')
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
    print(f'Model sparsity = {measure_sparsity(model)}')
    ckpt_path = callbacks[0].best_model_path
    model_checkpoint = torch.load(ckpt_path)
    model.load_state_dict(model_checkpoint["state_dict"])
    
    if args.pruning_method:
        remove_parameters(model)
    torch.save(
        model.state_dict(),
        f"{args.dirpath}/{args.experiment_name}/{args.experiment_name}.pt",
    )
