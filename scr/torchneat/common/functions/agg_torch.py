import torch

def sum_(z, dim=0):
    """
    Compute the sum along a dimension, ignoring NaNs.
    """
    z = torch.where(torch.isnan(z), torch.tensor(0.0, device=z.device), z)
    return torch.sum(z, dim=dim)

def product_(z, dim=0):
    """
    Compute the product along a dimension, ignoring NaNs.
    """
    z = torch.where(torch.isnan(z), torch.tensor(1.0, device=z.device), z)
    return torch.prod(z, dim=dim)

def max_(z, dim=0):
    """
    Compute the maximum along a dimension, ignoring NaNs.
    """
    z = torch.where(torch.isnan(z), torch.tensor(-float('inf'), device=z.device), z)
    return torch.max(z, dim=dim).values

def min_(z, dim=0):
    """
    Compute the minimum along a dimension, ignoring NaNs.
    """
    z = torch.where(torch.isnan(z), torch.tensor(float('inf'), device=z.device), z)
    return torch.min(z, dim=dim).values

def maxabs_(z, dim=0):
    """
    Compute the maximum absolute value along a dimension, ignoring NaNs.
    """
    z = torch.where(torch.isnan(z), torch.tensor(0.0, device=z.device), z)
    abs_z = torch.abs(z)
    max_abs_index = torch.argmax(abs_z, dim=dim)
    return torch.gather(z, dim, max_abs_index.unsqueeze(dim)).squeeze(dim)

def mean_(z, dim=0):
    """
    Compute the mean along a dimension, ignoring NaNs.
    """
    z = torch.where(torch.isnan(z), torch.tensor(0.0, device=z.device), z)
    valid_count = torch.sum(~torch.isnan(z), dim=dim)
    return torch.sum(z, dim=dim) / valid_count