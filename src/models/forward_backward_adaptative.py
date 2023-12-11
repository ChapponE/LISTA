import numpy as np
import torch
from src.utils.math_function import max_eig

class Forward_Backward_adaptative():
    def __init__(self, size_observed, size_target, A, lambd=0.2, Niter=16, device='cpu'):
        self.A = A
        gram_A, L = max_eig(self.A)
        self.gram_A = gram_A
        self.L = L * 1.001
        self.tau = 1 / L
        self.lambd = lambd
        self.Niter = Niter
        self.size_observed = size_observed
        self.size_target = size_target
        self.tol = 10 ** (-5)
        self.converged = 10 ** (-3)
        self.eps = 1.5
        self.device = device

    def __shrink__(self, x):
        return torch.sign(x) * (torch.abs(x) - self.lambd * self.tau)

    def iterations(self, x, y):
        u = torch.zeros((x.shape[0], self.size_target)).to(self.device)
        MSE_list = [0]
        for i in range(self.Niter):
            u0 = u
            u = self.__shrink__(u + torch.matmul(x - torch.matmul(u, self.A.transpose(0, 1)), self.A) / self.L)
            # adaptative part:
            norm_x = torch.norm(u - u0, p=2, dim=1)
            if i > 0:
                if norm_x < self.eps:
                    self.lambd = self.lambd * 0.5
                    self.eps = 0.5 * self.eps
        return u

    def __partial_forward__(self, x, y):
        u = torch.zeros((x.shape[0], self.size_target)).to(self.device)
        partials_u = [u.detach().clone()]
        partials_norm = []
        a = 0
        MSE_list = [0]
        for i in range(self.Niter):
            u = self.__shrink__(u + torch.matmul(x - torch.matmul(u, self.A.transpose(0, 1)), self.A) / self.L)
            partials_u.append(u)

            # adaptative part:
            # partials_norm += [torch.norm(partials_u[-2] - partials_u[-1], p=2)]
            # if a > 2:

            partial_error = partials_u[-1] - y
            norm_error = torch.norm(partial_error, p=2, dim=1) ** 2
            norm_x = torch.norm(x, p=2, dim=1) ** 2
            partials_norm += [10 * torch.log10(torch.mean(norm_error) / torch.mean(norm_x))]
            print('l49 adaptativ', partials_norm[-1], self.eps, self.lambd)
            a += 1
            if i > 1 and a > 20:
                if  torch.abs(partials_norm[-1]-partials_norm[-2]) < self.eps:
                    self.lambd = self.lambd * 0.6
                    # self.eps = torch.abs(partials_norm[-1]-partials_norm[-2])*10
                    self.eps = self.eps * 0.5
                    a = 0
                # else:
                #     self.lambd = self.lambd * 1.1
                #     self.eps = torch.abs(partials_norm[-1]-partials_norm[-2])/1.5

        return partials_u

    def __partial_NMSE__(self, x, y):
        partials_u = self.__partial_forward__(x, y)
        partials_NMSE = []
        for partial_u in partials_u:
            partial_error = partial_u - y
            norm_error = torch.norm(partial_error, p=2, dim=1) ** 2
            norm_x = torch.norm(x, p=2, dim=1) ** 2
            partials_NMSE.append(10 * torch.log10(torch.mean(norm_error) / torch.mean(norm_x)))
        return (partials_NMSE)
