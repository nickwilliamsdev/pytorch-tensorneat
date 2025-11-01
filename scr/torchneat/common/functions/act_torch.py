import torch

SCALE = 3


def scaled_sigmoid_(z):
    z = 1 / (1 + torch.exp(-z))
    return z * SCALE


def sigmoid_(z):
    z = 1 / (1 + torch.exp(-z))
    return z


def scaled_tanh_(z):
    return torch.tanh(z) * SCALE


def tanh_(z):
    return torch.tanh(z)


def sin_(z):
    return torch.sin(z)


def relu_(z):
    return torch.maximum(z, torch.tensor(0.0))


def lelu_(z):
    leaky = 0.005
    return torch.where(z > 0, z, leaky * z)


def identity_(z):
    return z


def inv_(z):
    # avoid division by zero
    z = torch.where(z > 0, torch.maximum(z, torch.tensor(1e-7)), torch.minimum(z, torch.tensor(-1e-7)))
    return 1 / z


def log_(z):
    z = torch.maximum(z, torch.tensor(1e-7))
    return torch.log(z)


def exp_(z):
    return torch.exp(z)


def abs_(z):
    return torch.abs(z)