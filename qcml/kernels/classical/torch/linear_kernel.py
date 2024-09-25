import torch

def linear_kernel(x1, x2):
    return torch.matmul(x1, x2)
