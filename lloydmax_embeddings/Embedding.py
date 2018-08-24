import torch
import torch.nn as nn
import bitarray as ba
import numpy as np
from .core import decode


class Embedding(nn.Module):

    def __init__(self, bit_arr, metadata):
        super(Embedding, self).__init__()
        self.meta = metadata
        self.bit_arr = bit_arr

    def forward(self, input):
        dim = self.meta['embed_dim']
        codes = self.meta['codebook']
        in_vect = input.flatten().data.numpy()
        tensor_shape = list(input.shape)
        tensor_shape.append(dim)
        embeds = [decode(i, self.bit_arr, self.meta) for i in in_vect]
        return torch.Tensor(np.vstack(embeds)).view(tensor_shape)

