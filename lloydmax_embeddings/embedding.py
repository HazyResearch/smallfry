import torch
import torch.nn as nn
import bitarray as ba
import numpy as np
from .smallfry import Smallfry


class SmallfryEmbedding(nn.Module):

    def __init__(self, sfry):
        super(SmallfryEmbedding, self).__init__()
        self.sfry = sfry

    def forward(self, input):
        print(input[0])
        print(type(input))
        a = input
        return torch.from_numpy(
            self.sfry.decode(input.to(device='cpu').numpy())).to(device=input.device)
