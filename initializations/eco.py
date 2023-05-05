from sao_utils.ramanujan_constructions import *
from .delta import *


class ECO_Module:
    def __init__(
        self,
        module: nn.Module,
        sparsity: float = None,
        degree: int = None,
        method: str = "SAO",
        activation: str = "tanh",
        in_channels: int = 3,
        num_classes: int = 10,
    ):
        self.module = module
        self.kernel_size = module.kernel_size[0]
        self.in_ch = module.in_channels
        self.out_ch = module.out_channels
        self.sparsity = sparsity
        self.degree = degree if self.sparsity is None else self._degree_from_sparsity()
        self.in_channels = in_channels  # Input channel of the model, not the module
        self.num_classes = num_classes
        self.method = method
        self.activation = activation

    def _ortho_gen(self, rows, columns) -> torch.tensor:
        rand_matrix = torch.randn((max(rows, columns), min(rows, columns)))
        q, _ = torch.qr(rand_matrix)
        orthogonal_matrix = q[:, :columns]
        return orthogonal_matrix.T if columns > rows else orthogonal_matrix

    def _concat(self, matrix) -> torch.tensor:
        W = torch.concat(
            [
                torch.concat([matrix, torch.negative(matrix)], axis=0),
                torch.concat([torch.negative(matrix), matrix], axis=0),
            ],
            axis=1,
        )
        return W

    def _ortho_generator(self) -> torch.tensor:
        if self.activation == "relu" and self.in_ch != 3:  # Input convolutional layer
            rows = self.out_ch // 2
            columns = self.in_ch // 2
            orthogonal_matrix = self._concat(self._ortho_gen(rows, columns))

        else:
            rows = self.out_ch
            columns = self.in_ch
            orthogonal_matrix = self._ortho_gen(rows, columns)

        return orthogonal_matrix

    def _degree_from_sparsity(self):
        larger_dim = max(self.in_ch, self.out_ch)
        return int((1 - self.sparsity) * larger_dim)

    def _unique_ortho_tensor(self) -> torch.tensor:
        L = (self.kernel_size**2 + 1) // 2
        ortho_tensor = torch.zeros(L, self.out_ch, self.in_ch)

        if self.degree is not None and self.in_ch > 3:
            constructor = Ramanujan_Constructions(
                self.module, degree=self.degree, activation=self.activation
            )

        for i in range(L):
            ortho_tensor[i] = (
                self._ortho_generator()
                if (self.degree is None or self.in_ch == 3)
                else constructor()[0]  # Get only the weights given by the constructor
            )

        return ortho_tensor.to("cuda")

    def _give_equiv(self, i: int, j: int):
        i_equiv = (self.kernel_size - i) % self.kernel_size
        j_equiv = (self.kernel_size - j) % self.kernel_size
        return i_equiv, j_equiv

    def _ortho_conv(self):
        k = self.kernel_size
        List1 = []
        List2 = []

        for i, j in product(range(k), range(k)):
            eqi, eqj = self._give_equiv(i, j)
            List1.append([i, j])
            List2.append([eqi, eqj])

        for i in List1:
            index1 = List1.index(i)
            index2 = List2.index(i)

            if index1 > index2:
                List1[index1] = -1

        List1 = [x for x in List1 if x != -1]
        List2 = [x for x in List2 if x not in List1]

        ortho_tensor = self._unique_ortho_tensor()
        A = torch.zeros(k, k, self.out_ch, self.in_ch)

        for i in range(len(List1)):
            p, q = List1[i]
            A[p, q] = ortho_tensor[i]

        for i in range(len(List2)):
            p, q = List2[i]
            equivi, equivj = self._give_equiv(p, q)
            A[p, q] = A[equivi, equivj]

        weight_mat = torch.zeros(self.out_ch, self.in_ch, k, k)

        for i, j in product(range(self.out_ch), range(self.in_ch)):
            weight_mat[i, j] = torch.fft.ifft2(A[:, :, i, j])

        return weight_mat.to("cuda")

    def __call__(self) -> torch.tensor:
        return self._ortho_conv()


def ECO_Constructor(module, **kwargs):
    return ECO_Module(module, **kwargs)()


def ECO_Init(model, gain, **kwargs):
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight = nn.Parameter(ECO_Constructor(module, **kwargs) * gain)
        elif isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 1)

    return model


def Delta_ECO_Init(model, gain, **kwargs):
    delta_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in ["activation", "in_channels", "num_classes"]
    }
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and (
            module.in_channels == 3 or module.stride[0] > 1
        ):
            module.weight = nn.Parameter(
                Delta_Constructor(module, **delta_kwargs) * gain
            )
        elif isinstance(module, nn.Conv2d) and (
            module.in_channels > 3 and module.stride[0] == 1
        ):
            module.weight = nn.Parameter(ECO_Constructor(module, **kwargs) * gain)
        elif isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 1)

    return model
