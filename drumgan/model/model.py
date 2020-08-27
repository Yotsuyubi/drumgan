import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import SamePaddingConv1d, DownsampleBlock, UpsampleBlock, FIRFilter



class Discriminator(nn.Module):

    def __init__(
        self,
        num_step=5,
        step=5,
        alpha=1,
        eval=False
    ):
        super().__init__()
        self.num_step = num_step
        self.step = step
        self.alpha = alpha
        self.eval = eval

        self.in_conv = SamePaddingConv1d(1, 64, 1)
        self.blocks = nn.ModuleList([DownsampleBlock(64, 64) for _ in range(self.num_step)])
        self.fc = nn.Linear(64*16, 1)

    def forward(self, x):
        if self.step >= 2 and self.alpha != 1 and self.eval == False:
            y = Resample(1/4)(x)
            x = self.in_conv(x)
            for i in range(self.num_step-self.step, self.num_step):
                x = self.blocks[i](x)
                if i == self.num_step-self.step:
                    x = self.alpha*x + (1-self.alpha)*y
        else:
            x = self.in_conv(x)
            for i in range(self.num_step-self.step, self.num_step):
                x = self.blocks[i](x)
        x = nn.Flatten()(x)
        return self.fc(x)

class SignalGenerator(nn.Module):

    def __init__(
        self,
        num_step=5,
        step=5,
        alpha=1,
        eval=False
    ):
        super().__init__()
        self.num_step = num_step
        self.step = step
        self.alpha = alpha
        self.evel = eval

        self.fc = nn.Linear(128, 64*16)
        self.blocks = nn.ModuleList([UpsampleBlock(64, 64) for _ in range(self.num_step)])
        self.out_conv = nn.Conv1d(64, 1, 1)

    def forward(self, noise):
        x = self.fc(noise)
        x = x.reshape(-1, 64, 16)
        if self.step >= 2 and self.alpha != 1 and eval == False:
            for i in range(self.step):
                if i == self.step-1:
                    y = Resample(4)(self.out_conv(x))
                x = self.blocks[i](x)
            x = self.out_conv(x)
            x = self.alpha*x + (1-self.alpha)*y
            return nn.Tanh()(x)
        else:
            for i in range(self.step):
                x = self.blocks[i](x)
            x = self.out_conv(x)
            return nn.Tanh()(x)


class GAN(nn.Module):

    def __init__(
        self,
        device='cpu',
    ):
        super().__init__()
        self.length = 16384
        self.latent_length = 128
        self.num_step = 5
        self.step = 5
        self.alpha = 1
        self.dev = device

        self.D = Discriminator(
            step=self.step,
            num_step=self.num_step,
            eval=self.eval,
            alpha=self.alpha
        )
        self.SigG = SignalGenerator(
            step=self.step,
            num_step=self.num_step,
            eval=self.eval,
            alpha=self.alpha
        )
        self.filter = FIRFilter(1, 1, 512)

    def forward(self, x):
        return self.filter(self.SigG(x.to(self.dev)))
