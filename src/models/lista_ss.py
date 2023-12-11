
import torch
from src.utils.math_function import max_eig
import torch.nn as nn

class LISTASS_model(torch.nn.Module):
    def __init__(self, size_observed, size_target, A, percent=1., max_percent=5, depth=16, init=True, device='cpu'):
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

        self.p = percent
        self.max_p = max_percent

        self.ps = [(t + 1) * self.p for t in range(self.depth)]
        self.ps = torch.clamp(torch.tensor(self.ps), 0.0, self.max_p).to(device)

        self.device = device

        # rates of Thresholds:
        if init:
            self.threshold = parameter.Parameter(torch.ones([self.depth], dtype=torch.float32).to(device) / self.L).to(
                device)
            # self.param_group = [{"params": self.threshold, "l1_proj": lambda x: x.clamp(min=0)}]
        else:
            self.threshold = parameter.Parameter(torch.randn(self.depth)).to(device)
            # self.param_groups = [{"params": self.threshold, "l1_proj": lambda x: x.clamp(min=0)}]

        # Weights and Biases:
        self.bias_layers = ModuleList()
        self.weight_layers = ModuleList()

        for i in range(depth):
            # Init weight_layers & bias_layers
            weight_layer = Linear(size_target, size_target, bias=False)
            bias_layer = Linear(size_observed, size_target, bias=False)

            if init:
                weight_layer.weight = nn.Parameter(torch.eye(self.size_target).to(device) - self.gram_A / self.L)
                bias_layer.weight = nn.Parameter(self.A.transpose(0, 1) / self.L)

            self.weight_layers.append(weight_layer.to(device))
            self.bias_layers.append(bias_layer.to(device))

    def __shrink__(self, x, threshold):
        # return torch.sign(x) * torch.max(torch.abs(x) - torch.max(threshold, torch.tensor([0.]).to(device)), torch.zeros_like(x))
        return torch.sign(x) * (torch.abs(x) - threshold).clamp(min=0.)

    def __shrinkss__(self, x, threshold, p):
        with torch.no_grad():
            abs_x = torch.abs(x).to(self.device)
            thresh_index = torch.ceil((100 - p) / 100 * x.shape[1]).to(self.device)
            sorted_rows, _ = torch.sort(abs_x)
            thresh = sorted_rows[:, int(thresh_index) - 1].to(self.device)
            index_ = (abs_x >= threshold) & (abs_x >= thresh[:, None])
            index_ = index_.float().detach()
            cindex_ = 1.0 - index_
        return (index_ * x + self.__shrink__(cindex_ * x, threshold))

        # abs_ = torch.abs(x).to(device)
        # thresh = torch.zeros(x.shape[0]).to(device)
        # with torch.no_grad():
        #   thresh_index = torch.ceil((100 - p) / 100 * x.shape[1]).to(device)
        #   for i in range(x.shape[0]):
        #       sorted_row, _ = torch.sort(x[i])
        #       thresh[int(i)] = sorted_row[(thresh_index.to(torch.long)).to(device)].to(device)

        #   index_ = (abs_ > threshold) & (abs_ > thresh[:,None])
        #   index_ = index_.float()
        #   index_ = index_.detach()
        #   cindex_ = 1.0 - index_  # complementary index
        # # Il faut enlever index_ et cindex_ dans le return
        # # return (index_ * x +
        # #         self.__shrink__(cindex_ * x, threshold), index_, cindex_)
        # return (index_ * x +
        #         self.__shrink__(cindex_ * x, threshold))

    def forward(self, x):

        u = torch.zeros((x.shape[0], self.size_target)).to(self.device)

        for weight, bias, i in zip(self.weight_layers, self.bias_layers, range(self.depth)):
            u = self.__shrinkss__(weight(u) + bias(x), self.threshold[i], self.ps[i])
        return u

    def __partial_forward__(self, x):
        u = torch.zeros((x.shape[0], self.size_target)).to(self.device)
        partials_u = [u]
        for weight, bias, i in zip(self.weight_layers, self.bias_layers, range(self.depth)):
            u = self.__shrinkss__(weight(u) + bias(x), self.threshold[i], self.ps[i])
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

