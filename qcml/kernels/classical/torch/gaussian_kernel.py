import torch

def gaussian_kernel(x1, x2, gamma=1):
    return torch.exp(-gamma * torch.sum((x1 - x2) ** 2))
