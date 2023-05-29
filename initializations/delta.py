import torch
import torch.nn as nn
from itertools import product
from .ramanujan_constructions import Ramanujan_Constructions
from .base import Base


class Delta_Module(Ramanujan_Constructions, Base):
    def __init__(
        self,
        module: nn.Module,
        gain: int = 1,
        sparsity: float = None,
        degree: int = None,
        method: str = "SAO",
        activation: str = "relu",
        same_mask: bool = False,
        in_channels: int = 3,
        num_classes: int = 100,
    ):
        self.module = module
        self.sparsity = sparsity
        self.degree = degree
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.method = method
        self.activation = activation
        self.same_mask = same_mask
        self.gain = gain

    def _sao_linear(self):
        constructor = self._ramanujan_structure()
        return constructor()

    def _sao_delta(self):
        constructor = self._ramanujan_structure()
        sao_matrix, sao_mask = constructor
        sao_delta_weights = torch.zeros_like(self.module.weight).to("cuda")
        sao_delta_weights[:, :, 1, 1] = sao_matrix
        sao_delta_mask = torch.zeros_like(self.module.weight).to("cuda")

        for i, j in product(
            range(self.module.out_channels), range(self.module.in_channels)
        ):
            sao_delta_mask[i, j] = sao_mask[i, j]

        return sao_delta_weights, sao_delta_mask

    def _delta(self):
        weights = self._ortho_generator()
        delta_weights = torch.zeros_like(self.module.weight).to("cuda")
        delta_weights[:, :, 1, 1] = weights

        return delta_weights

    def __call__(self):
        return (
            self._sao_delta()
            if (self.degree or self.sparsity)
            and self.module.in_channels > 3
            and self.method == "SAO"
            else self._delta()
        )


def Delta_Constructor(module, **kwargs):
    return Delta_Module(module, **kwargs)()
