import torch
import torch.nn as nn
from itertools import product
from sao_utils.ramanujan_constructions import Ramanujan_Constructions


class Delta_Module(Ramanujan_Constructions):
    def __init__(
        self,
        module: nn.Module,
        gain: int = 1,
        sparsity: float = None,
        degree: int = None,
        method: str = "SAO",
        activation: str ="relu",
        same_mask: bool=False,
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
        
    def _ortho_gen(self, rows, columns) -> torch.tensor:
        rand_matrix = torch.randn((max(rows, columns), min(rows, columns)))
        q, _ = torch.qr(rand_matrix)
        orthogonal_matrix = q[:, :columns]
        return orthogonal_matrix.T if columns > rows else orthogonal_matrix   
        
        
    def _ramanujan_structure(self):
        constructor = Ramanujan_Constructions(
            self.module,
            sparsity=self.sparsity,
            degree=self.degree,
            method=self.method,
            same_mask=self.same_mask,
            activation=self.activation,
        )
        return constructor()
    
    
    def _sao_linear(self):
        return self._ramanujan_structure()


    def _sao_delta(self):
        sao_matrix, sao_mask = self._ramanujan_structure()
        sao_delta_weights = torch.zeros_like(self.module.weight).to("cuda")
        sao_delta_weights[:, :, 1, 1] = sao_matrix
        sao_delta_mask = torch.zeros_like(self.module.weight).to("cuda")

        for i, j in product(range(self.module.out_channels), range(self.module.in_channels)):
            sao_delta_mask[i, j] = sao_mask[i, j]

        return sao_delta_weights, sao_delta_mask
    
    def _delta(self):
        if self.activation == 'relu' or self.module.in_channels == 3:
            W_0 = self._ortho_gen(
                    self.module.weight.shape[0] // 2, self.module.weight.shape[1] // 2
                )
            weights = self._relu_util(W_0)
    
        else:
            weights = self._ortho_gen(
                    self.module.weight.shape[0], self.module.weight.shape[1])
        delta_weights = torch.zeros_like(self.module.weight).to("cuda")
        delta_weights[:, :, 1, 1] = weights

        return delta_weights
    
    def __call__(self):
        return self._sao_delta() if self.degree or self.sparsity else self._delta()


def Delta_Constructor(module, **kwargs):
    return Delta_Module(module, **kwargs)()

