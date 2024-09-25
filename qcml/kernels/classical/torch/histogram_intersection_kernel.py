import torch

def histogram_intersection_kernel(x1, x2):
    return torch.sum(torch.min(x1, x2))
