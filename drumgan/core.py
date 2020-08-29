from .model import GAN
from .helpers import to_tensor, download_model
import torch
import numpy as np
from typing import Union
import os



def input_to_tensor_output_to_numpy(func):
    def wrapper(*args, **kwargs):
        args = list(args)
        args[1] = to_tensor(args[1])
        res = func(*args, **kwargs)
        if len(res) > 1:
            return (r.detach().numpy() for r in res)
        else:
            return res.detach().numpy()
    return wrapper


class DrumGAN():

    def __init__(self):
        download_model()
        self.model = GAN(device='cpu')
        self.model.load_state_dict(
            torch.load(
                os.path.join(os.path.dirname(__file__), 'model/model.pt'),
                map_location='cpu'
            ),
            strict=False
        )

    @input_to_tensor_output_to_numpy
    def generate(
        self,
        z: Union[torch.Tensor, np.ndarray, list]
    ) -> (np.ndarray, np.ndarray):
        assert (z.shape == (1, 128))
        return self.model(z)[0][0], z

    def random_generate(self) -> (np.ndarray, np.ndarray):
        z = torch.rand([1, 128])
        return self.generate(z)

    @input_to_tensor_output_to_numpy
    def optim_feature(
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
