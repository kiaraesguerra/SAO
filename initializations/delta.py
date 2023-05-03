import torch
import torch.nn as nn


class Delta_Module:
    def __init__(
        self,
        module: nn.Module,
        activation: str = "tanh",
        in_channels: int = 3,
        num_classes: int = 10,
    ) -> nn.Module:
        self.module = module
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.activation = activation

    def _ortho_gen(self, rows, columns) -> torch.tensor:
        rand_matrix = torch.randn((max(rows, columns), min(rows, columns)))
        q, _ = torch.qr(rand_matrix)
        orthogonal_matrix = q[:, :columns]
        return orthogonal_matrix.T if columns > rows else orthogonal_matrix

    def _delta(self) -> torch.tensor:
        if (
            self.activation == "relu"
            and self.module.weight.shape[1] != 3
            and self.module.weight.shape[0] != 10
        ):
            print(self.module.weight.shape, self.module.weight.shape[0])
            W_0 = self._ortho_gen(
                self.module.weight.shape[0] // 2, self.module.weight.shape[1] // 2
            )
            delta_matrix = torch.concat(
                [
                    torch.concat([W_0, torch.negative(W_0)], axis=0),
                    torch.concat([torch.negative(W_0), W_0], axis=0),
                ],
                axis=1,
            )
        else:
            delta_matrix = self._ortho_gen(
                self.module.weight.shape[0], self.module.weight.shape[1]
            )

        if isinstance(self.module, nn.Conv2d):
            delta_weights = torch.zeros_like(self.module.weight).to("cuda")
            delta_weights[:, :, 1, 1] = delta_matrix
        else:
            delta_weights = delta_matrix

        return delta_weights

    def __call__(self):
        return self._delta()


def Delta_Constructor(module, activation="tanh", in_channels=3, num_classes=10):
    return Delta_Module(
        module, activation=activation, in_channels=in_channels, num_classes=num_classes
    )()
