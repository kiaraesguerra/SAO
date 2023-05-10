import torch
import torch.nn as nn
import random
from scipy.linalg import orth
from itertools import product
import numpy as np


class Ramanujan_Constructions:
    def __init__(
        self,
        module: nn.Module,
        sparsity: float = None,
        degree: int = None,
        method: str = "SAO",
        same_mask: bool = True,
        activation: str = "tanh",
        device: str = "cuda",
    ):
        """_summary_

        Args:
            module (nn.Module): The module to be processed
            degree (int): _description_
            mode (str, optional): _description_. Defaults to "O".
            activation (str, optional): _description_. Defaults to "tanh".
            device (str, optional): _description_. Defaults to "cuda".
        """

        self.in_ = module.weight.shape[1]
        self.out_ = module.weight.shape[0]
        self.rows = min(self.out_, self.in_)
        self.columns = max(self.out_, self.out_)
        self.sparsity = sparsity
        self.degree = degree if sparsity is None else self._degree_from_sparsity()
        self.method = method
        self.device = device
        self.activation = activation
        self.same_mask = same_mask
        self.ramanujan_mask = None

        if (
            self.activation == "relu"
            and self.in_ != 3  # Input convolutional layer
            and self.out_ != 100  # Output linear layer
        ):
            self.rows = self.rows // 2
            self.columns = self.columns // 2
            self.degree = self.degree // 2

    def _degree_from_sparsity(self):
        larger_dim = max(self.in_, self.out_)
        return int((1 - self.sparsity) * larger_dim)

    def _relu_util(self, matrix):
        W = torch.concat(
            [
                torch.concat([matrix, torch.negative(matrix)], axis=0),
                torch.concat([torch.negative(matrix), matrix], axis=0),
            ],
            axis=1,
        )

        return W

    def _block_construct(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        block_rows = int(self.columns / self.degree)
        block_columns = self.columns
        d_vector = (1 / 2) * torch.ones(self.columns)
        target_d_vector = torch.ones(self.columns)
        while not torch.equal(d_vector, target_d_vector):
            block = torch.zeros(block_rows, block_columns)
            columns = list(range(block_columns))
            for _ in range(2):
                for i in range(block.shape[0]):
                    index = random.choice(columns)
                    while torch.sum(block[:, index]) == 1:
                        index = random.choice(columns)
                    block[i, index] = 1

            for i, j in product(range(block_rows), range(block_columns)):
                if (
                    block[i, j] != 1
                    and torch.sum(block[i]) < self.degree
                    and torch.sum(block[:, j]) != 1
                ):
                    block[i, j] = 1
                d_vector = torch.sum(block, 0)
        mask = torch.zeros(self.rows, self.columns)
        a = int((self.rows * self.degree) / self.columns)
        for i in range(a):
            mask[block_rows * i : block_rows * i + block_rows] = block.reshape(
                block_rows, block_columns
            )
        return torch.tensor(mask).reshape(self.rows, self.columns).to(self.device)

    def _val_generator(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if self.method == "SAO":
            M = np.random.rand(self.degree, self.degree)
            return torch.tensor(orth(M), dtype=torch.float).to(self.device)
        elif self.method == "RG-N":
            return torch.randn(self.degree, self.degree).to(self.device)
        elif self.method == "RG-U":
            return torch.rand(self.degree, self.degree).to(self.device)

    def _assign_values(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # if self.ramanujan_mask is not None:
        #     ramanujan_mask = self.ramanujan_mask
        # else:
        ramanujan_mask = self._block_construct()
        #ramanujan_mask = torch.abs(ramanujan_mask)
        self.ramanujan_mask = ramanujan_mask

        c = int(torch.sum(ramanujan_mask, 0)[0])
        d = int(torch.sum(ramanujan_mask, 1)[0])
        degree = c if c > d else d
        sao_matrix = torch.zeros(ramanujan_mask.shape).to(self.device)
        num_ortho = int(degree * ramanujan_mask.shape[0] / ramanujan_mask.shape[1])

        _, inv, counts = torch.unique(
            ramanujan_mask, dim=0, return_inverse=True, return_counts=True
        )
        row_index = [
            tuple(torch.where(inv == i)[0].tolist())
            for i, c, in enumerate(counts)
            if counts[i] > 1
        ]

        if num_ortho == 1:
            to_iterate = inv.reshape(inv.shape[0], 1)
        else:
            to_iterate = row_index

        for i in to_iterate:
            indices = torch.tensor(i).to(self.device)
            identical_row = ramanujan_mask[indices]
            vals = self._val_generator()
            for j in range(identical_row.shape[0]):
                nonzeros = torch.nonzero(identical_row[j])
                identical_row[j, nonzeros] = (
                    vals[j].reshape(vals.shape[1], 1).to(self.device)
                )
            sao_matrix[indices] = identical_row

        return sao_matrix

    def __call__(self):
        weights = (
            self._relu_util(self._assign_values())
            if self.activation == "relu"
            else self._assign_values()
        )
        mask = (
            self._relu_util(self.ramanujan_mask)
            if self.activation == "relu"
            else self.ramanujan_mask
        )
        return (weights.T, mask.T) if self.out_ > self.in_ else (weights, mask)


class Ramanujan_Construction:
    def __init__(
        self,
        model,
        gain: int = 1,
        sparsity: float = None,
        degree: int = None,
        method: str = "SAO",
        activation="relu",
        same_mask=False,
        in_channels: int = 3,
        num_classes: int = 100,
    ):
        """_summary_

        Args:
            model (_type_): _description_
            degree (int): _description_
            mode (str, optional): _description_. Defaults to "O".
            num_classes (int, optional): _description_. Defaults to 10.
        """
        self.model = model
        self.sparsity = sparsity
        self.degree = degree
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.method = method
        self.activation = activation
        self.same_mask = same_mask
        self.gain = gain

    def _ramanujan_structure(self, module):
        constructor = Ramanujan_Constructions(
            module,
            sparsity=self.sparsity,
            degree=self.degree,
            method=self.method,
            same_mask=self.same_mask,
            activation=self.activation,
        )
        return constructor()

    def _sao_linear(self, module):
        return self._ramanujan_structure(module)

    def _sao_delta(self, module):
        sao_matrix, sao_mask = self._ramanujan_structure(module)
        sao_delta_weights = torch.zeros_like(module.weight).to("cuda")
        sao_delta_weights[:, :, 1, 1] = sao_matrix
        sao_delta_mask = torch.zeros_like(module.weight).to("cuda")

        for i, j in product(range(module.out_channels), range(module.in_channels)):
            sao_delta_mask[i, j] = sao_mask[i, j]

        return sao_delta_weights, sao_delta_mask

    def _sao_init(self):
        for _, module in self.model.named_modules():
            if (
                isinstance(module, nn.Linear)
                and module.out_features != self.num_classes
            ):
                weight, mask = self._sao_linear(module)
                mask = torch.abs(mask)
                module.weight = nn.Parameter(weight * self.gain)
                torch.nn.utils.prune.custom_from_mask(module, "weight", mask)

            elif isinstance(module, nn.Conv2d) and module.in_channels != 3:
                weight, mask = self._sao_delta(module)
                mask = torch.abs(mask)
                module.weight = nn.Parameter(weight * self.gain)
                torch.nn.utils.prune.custom_from_mask(module, "weight", mask)

        return self.model

    def __call__(self):
        return self._sao_init()
