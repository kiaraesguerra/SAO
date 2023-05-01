import torch
import torch.nn as nn
import math
from itertools import product


def prune_vanilla_kernelwise(param, sparsity, fn_importance=lambda x: x.norm(1, -1)):
    assert param.dim() >= 3
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        return torch.zeros_like(param).byte()
    num_kernels = param.size(0) * param.size(1)
    param_k = param.view(num_kernels, -1)
    param_importance = fn_importance(param_k)
    num_pruned = int(math.ceil(num_kernels * sparsity))
    _, topk_indices = torch.topk(
        param_importance, k=num_pruned, dim=0, largest=False, sorted=False
    )
    mask = torch.zeros_like(param).byte()
    mask_k = mask.view(num_kernels, -1)
    param_k.index_fill_(0, topk_indices, 0)
    mask_k.index_fill_(0, topk_indices, 1)
    return (mask == 0) * 1


def mask_random(weight, sparsity):
    m = weight.shape[0]
    n = weight.shape[1]

    mask_tensor = torch.zeros(weight.shape)

    prob = torch.rand(m * n)
    elements_to_prune = sparsity * m * n
    threshold = torch.kthvalue(prob, k=int(elements_to_prune)).values
    channel_mask = ((prob > threshold) * 1).reshape(m, n)

    for i, j in product(range(m), range(n)):
        mask_tensor[i, j] = channel_mask[i, j]

    return mask_tensor.to("cuda")


class Standard_Pruning:
    def __init__(
        self,
        model,
        pruner,
        sparsity: float = None,
        degree: int = None,
        in_channels: int = 3,
        num_classes: int = 10,
    ):
        self.model = model
        self.degree = degree
        self.sparsity = sparsity if degree is None else self._sparsity_from_degree()
        self.in_channels = in_channels
        self.num_classes = num_classes

        if pruner == "lrp":
            self.pruner = mask_random
        elif pruner == "lmp":
            self.pruner = prune_vanilla_kernelwise

    def _sparsity_from_degree(self):
        return 1 - self.degree / max(self.in_, self.out_)

    def _pruner(self):
        for _, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels != self.in_channels:
                with torch.no_grad():
                    mask = self.pruner(module.weight, self.sparsity)
                torch.nn.utils.prune.custom_from_mask(module, name="weight", mask=mask)

        return self.model

    def __call__(self):
        return self._pruner()
