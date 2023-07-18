import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LowRankLinear, self).__init__()
        self.rank = rank
        self.U_layer = nn.Linear(in_features=rank, out_features=out_features)
        self.W_layer = nn.Linear(in_features=out_features, out_features=rank)

    def forward(self, x):
        x = self.U_layer(x)
        x = self.W_layer(x)
        return x

class LowRankInitializer:
    def __init__(self, model, rank):
        self.model = model
        self.rank = rank

    def low_rank(self, module):
        u, s, v = torch.linalg.svd(module.weight)
        padded_s = torch.zeros_like(module.weight)
        s_diag = torch.diag_embed(s)
        padded_s[0:s_diag.shape[0], 0:s_diag.shape[1]] = s_diag
        W_weight_matrix = torch.matmul(padded_s[0:self.rank, 0:self.rank], v[0:self.rank])
        U_weight_matrix = u[:, 0:self.rank]
        LR_module = LowRankLinear(module.in_features, module.out_features, self.rank)
        LR_module.U_layer.weight = nn.Parameter(U_weight_matrix)
        LR_module.W_layer.weight = nn.Parameter(W_weight_matrix)
   
        return LR_module

    def initialize_low_rank(self):
        for module_name, module in self.model.hidden_layers.named_modules():
            if isinstance(module, nn.Linear):
                self.model.hidden_layers._modules[module_name] = self.low_rank(module)