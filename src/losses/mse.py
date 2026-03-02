import torch.nn as nn


def get_mse_loss():
    """
    Returns standard MSE loss for density map regression.
    """
    return nn.MSELoss()