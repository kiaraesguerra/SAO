import torch
import torch.nn as nn
from itertools import product
from .ramanujan_constructions import Ramanujan_Constructions
from .base import Base


class LS_Module(Ramanujan_Constructions, Base):
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
        sao_matrix, sao_mask = constructor()

        return sao_matrix, sao_mask

    def __call__(self):
        return (
            self._sao_linear()
        )


def LS_Constructor(module, **kwargs):
    return LS_Module(module, **kwargs)()
