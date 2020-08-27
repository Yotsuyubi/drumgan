from .model.model import GAN
from .helpers import to_tensor
import torch
import numpy as np


def input_to_tensor_output_to_numpy(func):
    def wrapper(*args, **kwargs):
        args = list(args)
        args[1] = to_tensor(args[1])
        res = func(*args, **kwargs)
        return res.detach().numpy()[0][0]
    return wrapper


class DrumGAN():

    def __init__(self, device='cpu'):
        self.model = GAN(device=device)
        self.model.load_state_dict(
            torch.load('drumgan/model/model.pt', map_location=device),
            strict=False
        )

    @input_to_tensor_output_to_numpy
    def generate(self, z):
        assert (z.shape == (1, 128))
        return self.model(z)

    def random_generate(self):
        z = torch.rand([1, 128])
        return self.generate(z)
