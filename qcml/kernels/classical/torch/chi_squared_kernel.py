import torch

def chi_squared_kernel(x1, x2):
    return torch.sum((2 * x1 * x2) / (x1 + x2 + 1e-8))
