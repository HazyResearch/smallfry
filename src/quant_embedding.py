import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("quant embedding test")

LONG_BITS=64

def compress_long_vec(long_tensor, nbit):
    """
    we assume a single vector is along the last dimension.
    We compress 
    """
    assert long_tensor.dtype == torch.int64
    assert LONG_BITS % nbit == 0
    # n_entry is the # of entries each long value can contain
    n_entry = LONG_BITS // nbit
    mask = int("".join(['0'] * (LONG_BITS - nbit) + ['1'] * nbit), 2)
    out_shape = list(long_tensor.shape)
    out_shape[-1] = math.ceil(out_shape[-1] / n_entry)
    out = torch.zeros(*out_shape, device=long_tensor.device, dtype=torch.int64)
    out_flat = out.view(-1, out_shape[-1])
    long_tensor_flat = long_tensor.view(-1, long_tensor.shape[-1])

    for i in range(n_entry):
        # number of value to compress
        n_val = long_tensor_flat[:, i::n_entry].size(-1)
        out_flat[:, :n_val] |= (long_tensor_flat[:, i::n_entry] & mask) << ((n_entry - i - 1) * nbit)
    return out # out is the original version of out_flat


def decompress_long_vec(byte_tensor, nbit, dim=None):
    """
    we assume a single vector is along the last dimension.
    """
    assert byte_tensor.dtype == torch.int64
    assert LONG_BITS % nbit == 0
    n_entry = LONG_BITS // nbit
    mask = int("".join(['0'] * (LONG_BITS - nbit) + ['1'] * nbit), 2)
    out_shape = list(byte_tensor.shape)
    out_shape[-1] *= n_entry
    out = torch.zeros(*out_shape, device=byte_tensor.device, dtype=torch.int64)
    # manipulate as 2d tensors
    out_flat = out.view(-1, out_shape[-1])
    byte_tensor_flat = byte_tensor.view(-1, byte_tensor.shape[-1])
    for i in range(n_entry):
        out_flat[:, i::n_entry] = (byte_tensor_flat >> ( (n_entry - i - 1) * nbit)) & mask
    if dim is not None:
        # cut the redundent dimensions
        out_flat = out_flat[:, :dim].contiguous()
        out_shape = list(byte_tensor.shape)
        out_shape[-1] = dim
        out = out_flat.view(*out_shape)
    return out


"""
TODO: Jian. 
1 interface:
a. split nbit infer and value list construction into one function
b. make the underlying tensor compression into another function
c. for a and b, implement both for the tensor and file interface.
d. If necessary, can switch to int32 underlying representation. But not necessary now.
e. Test on GPU
"""

class QuantEmbedding(nn.Embedding):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.,
                 scale_grad_by_freq=False,
                 sparse=False,
                 _weight=None,
                 nbit=32,
                 embedding_file=None):
        """
        Impelmentation of the quantized embedding layer. This layer
        memory efficient embedding storage during inference. Currently,
        the implementation support 1, 2, 4, 8, 16 bit represention.
        The QuantEmbedding layer save the quantized representation in
        IntTensor, during forward for inference, the bits are extracted
        from IntTensor and put into Float32 tensor for inference.
        The quantized value are saved in text format. For reference format,
        please refer to http://nlp.stanford.edu/data/glove.6B.zip
        """
        assert nbit in (1, 2, 4, 8, 16, 32)
        assert max_norm == None
        assert norm_type == 2.
        assert scale_grad_by_freq == False
        assert sparse == False
        self.nbit = nbit
        if self.nbit == 32:
            self.tensor_dim = embedding_dim
        else:
            self.tensor_dim = math.ceil(embedding_dim * nbit / LONG_BITS)
            # weight = torch.zeros(
            #     num_embeddings, self.tensor_dim, dtype=torch.int32)
        nn.Embedding.__init__(
            self,
            num_embeddings, # we use the actual tensor dim here, otherwise will raise error
            self.tensor_dim,
            padding_idx,
            max_norm=None,
            norm_type=2.,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None)

        if self.nbit != 32:
            self.weight = nn.Parameter(torch.zeros(
                num_embeddings, self.tensor_dim, dtype=torch.int64), requires_grad=False)
        else:
            # we only support forward pass
            self.weight.requires_grad = False
        # recover the true embeding_dim
        self.embedding_dim = embedding_dim
        # load the quantized values from file to the int32 tensor
        self.load_from_file(embedding_file)

    def load_from_file(self, file_name):
        if self.nbit != 32:
            self.value_set = set([])
            # construct the mapping between quantized index and quantized value
            with open(file_name, "r") as f:
                for line_id, line in enumerate(f.readlines()):
                    for value in line.strip('\n').split(" ")[1:]:
                        self.value_set.add(float(value))
            sorted_vals = sorted(list(self.value_set))
            self.register_buffer(
                "value_list", torch.FloatTensor(sorted_vals))
            self.value_dict = {
                float(value): i
                for i, value in enumerate(sorted_vals)
            }
            line_cnt = line_id + 1
            assert len(self.value_set) <= 2**self.nbit
            if len(self.value_set) < 2**self.nbit:
                logger.warning("Set of actual values is smaller than set of possible values.")
        else:
            with open(file_name, "r") as f:
                line_cnt = len(f.readlines())
        assert self.num_embeddings == line_cnt

        # put vectors into int32 tensor
        with open(file_name, "r") as f:
            for line_id, line in enumerate(f.readlines()):
                if self.nbit != 32:
                    vector = torch.LongTensor([
                        self.value_dict[float(value)]
                        for value in line.strip('\n').split(" ")[1:]
                    ])
                    self.weight[line_id].copy_(compress_long_vec(vector, self.nbit))
                else:
                    self.weight[line_id].copy_(
                        torch.tensor(
                            [float(value) for value in line.split(" ")[1:]],
                            dtype=self.weight.dtype))

    def forward(self, input):
        embedding = F.embedding(input, self.weight, self.padding_idx,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, self.sparse)
        if self.nbit != 32:
            embedding = decompress_long_vec(embedding, self.nbit, self.embedding_dim)  
            embedding = self.value_list[embedding]
        return embedding



