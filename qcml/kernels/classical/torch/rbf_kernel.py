import torch

def rbf_kernel(x1, x2, gamma=None):
    """
    Compute the RBF kernel using PyTorch.

    Args:
        x1 (torch.Tensor): First vector with shape (n_features,).
        x2 (torch.Tensor): Second vector with shape (n_features,).
        gamma (float): Scaling parameter.

    Returns:
        torch.Tensor: Kernel value.
    """
    if gamma is None:
        gamma = 1.0 / x1.shape[1]
    sq_dist = torch.sum((x1[:, None] - x2[None, :]) ** 2, dim=-1)
    return torch.exp(-gamma * sq_dist)
