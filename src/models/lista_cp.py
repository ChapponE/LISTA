
import torch
from src.utils.math_function import max_eig
import torch.nn as nn

class LISTACP_model(torch.nn.Module):
    def __init__(self, size_observed, size_target, A, depth=16, init=True, device='cpu'):
        super().__init__()
        from torch.nn import ModuleList, Flatten, Linear, ReLU, parameter
        # Quelques attributs:
        self.A = A
        gram_A, L = max_eig(self.A)
        self.gram_A = gram_A
        self.L = L * 1.001

        self.depth = depth
        self.size_observed = size_observed
        self.size_target = size_target
        self.device = device

        # rates of Thresholds:
        if init:
            self.threshold = parameter.Parameter(torch.ones([self.depth], dtype=torch.float32).to(device) / self.L).to(
                device)
            # self.param_group = [{"params": self.threshold, "l1_proj": lambda x: x.clamp(min=0)}]
        else:
            self.threshold = parameter.Parameter(torch.randn(self.depth))
            # self.param_groups = [{"params": self.threshold, "l1_proj": lambda x: x.clamp(min=0)}]

        # Weights and Biases:
        self.bias_layers = ModuleList()
        for i in range(depth):
            # Init weight_layers & bias_layers
            bias_layer = Linear(size_observed, size_target, bias=False)

            if init:
                bias_layer.weight = nn.Parameter(self.A.transpose(0, 1) / self.L)

            self.bias_layers.append(bias_layer.to(device))

    def __shrink__(self, x, threshold):
        # return torch.sign(x) * torch.max(torch.abs(x) - torch.max(threshold, torch.tensor([0.]).to(device)), torch.zeros_like(x))
        return torch.sign(x) * (torch.abs(x) - threshold).clamp(min=0.)

    def forward(self, x):

        u = torch.zeros((x.shape[0], self.size_target)).to(self.device)

        for bias, i in zip(self.bias_layers, range(self.depth)):
            u = self.__shrink__(u + bias(x - torch.matmul(u, self.A.transpose(0, 1))), self.threshold[i])
        return (u)

    def __partial_forward__(self, x):
        u = torch.zeros((x.shape[0], self.size_target)).to(self.device)
        partials_u = [u]
        for bias, i in zip(self.bias_layers, range(self.depth)):
            u = self.__shrink__(u + bias(x - torch.matmul(u, self.A.transpose(0, 1))), self.threshold[i])
            partials_u.append(u)
        return partials_u

    def __partial_NMSE__(self, x, y):
        partials_u = self.__partial_forward__(x)
        partials_NMSE = []
        for partial_u in partials_u:
            partial_error = partial_u - y
            norm_error = torch.norm(partial_error, p=2, dim=1) ** 2
            norm_x = torch.norm(x, p=2, dim=1) ** 2
            partials_NMSE.append(10 * torch.log10(torch.mean(norm_error) / torch.mean(norm_x)))
        return (partials_NMSE)
