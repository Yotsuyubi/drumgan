import torch


def to_tensor(x):
    """Make x to Tensor."""
    try:
        return x.clone().detach().float()
    except:
        return torch.tensor(x).float()
