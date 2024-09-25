import torch

def sigmoid_kernel(x1, x2, alpha=1, c=0):
    return torch.tanh(alpha * torch.matmul(x1, x2) + c)
