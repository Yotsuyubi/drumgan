from .model import GAN
from .helpers import to_tensor
import torch
import numpy as np
from typing import Union


def input_to_tensor_output_to_numpy(func):
    def wrapper(*args, **kwargs):
        args = list(args)
        args[1] = to_tensor(args[1])
        res = func(*args, **kwargs)
        return res.detach().numpy()
    return wrapper


class DrumGAN():

    def __init__(self, device='cpu'):
        self.model = GAN(device=device)
        self.model.load_state_dict(
            torch.load('drumgan/model/model.pt', map_location=device),
            strict=False
        )

    @input_to_tensor_output_to_numpy
    def generate(
        self,
        z: Union[torch.Tensor, np.ndarray, list]
    ) -> np.ndarray:
        assert (z.shape == (1, 128))
        return self.model(z)[0][0]

    def random_generate(self) -> np.ndarray:
        z = torch.rand([1, 128])
        return self.generate(z)

    @input_to_tensor_output_to_numpy
    def train_feature(
        self,
        y: Union[torch.Tensor, np.ndarray, list],
        iteration: int = 500
    ) -> np.ndarray:
        assert (y.shape == (1, 1, 16384))
        z = torch.rand([1, 128])
        optim = torch.optim.Adam([z], lr=1e-3)
        for i in range(iteration):
            optim.zero_grad()
            y_hat = self.model(z)
            loss = torch.nn.MSELoss()(y_hat, y)
            loss.backward()
            optim.step()
        return z
