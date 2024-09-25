import torch

def polynomial_kernel(x1, x2, degree=3, coef0=1):
    return (torch.matmul(x1, x2) + coef0) ** degree
