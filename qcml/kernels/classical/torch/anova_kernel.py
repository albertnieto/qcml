import torch

def anova_kernel(x1, x2, d=2, sigma=1):
    return torch.sum(torch.exp(-sigma * (x1 - x2) ** 2)) ** d
