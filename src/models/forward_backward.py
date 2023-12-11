import numpy as np
import torch
from src.utils.math_function import max_eig

class Forward_Backward():
    def __init__(self, size_observed, size_target, A, lambd=0.1, Niter=16, device='cpu'):
        self.A = A
        gram_A, L = max_eig(self.A)
        self.gram_A = gram_A
        self.L = L * 1.001
        self.tau = 1 / L
        self.lambd = lambd
        self.Niter = Niter
        self.size_observed = size_observed
        self.size_target = size_target
        self.device = device

    def __shrink__(self, x):
        return torch.sign(x) * (torch.abs(x) - self.lambd * self.tau)

    def iterations(self, x):
        u = torch.zeros((x.shape[0], self.size_target)).to(self.device)
        for i in range(self.Niter):
            u = self.__shrink__(u + torch.matmul(x - torch.matmul(u, self.A.transpose(0, 1)), self.A) / self.L)
        return u

    def __partial_forward__(self, x):
        u = torch.zeros((x.shape[0], self.size_target)).to(self.device)
        partials_u = [u]
        for i in range(self.Niter):
            u = self.__shrink__(u + torch.matmul(x - torch.matmul(u, self.A.transpose(0, 1)), self.A) / self.L)
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
