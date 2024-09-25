import torch

def laplacian_kernel(x1, x2, gamma=1):
    return torch.exp(-gamma * torch.sum(torch.abs(x1 - x2)))
