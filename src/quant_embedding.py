import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import compress
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("quant embedding test")

LONG_BITS = 64


def fix_randomness(seed):
    np.random.seed(seed)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compress_long_mat(long_tensor, nbit):
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
        out_flat[:, :n_val] |= (long_tensor_flat[:, i::n_entry] & mask) << (
            (n_entry - i - 1) * nbit)
    return out  # out is the original version of out_flat


def decompress_long_mat(byte_tensor, nbit, dim=None):
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
        out_flat[:, i::n_entry] = (byte_tensor_flat >>
                                   ((n_entry - i - 1) * nbit)) & mask
    if dim is not None:
        # cut the redundent dimensions
        out_flat = out_flat[:, :dim].contiguous()
        out_shape = list(byte_tensor.shape)
        out_shape[-1] = dim
        out = out_flat.view(*out_shape)
    return out


def line2vec(line, value_dict=None):
    """
    convert a line in embedding file to a float tensor vector
    """
    if value_dict is None:
        return torch.FloatTensor(
            [float(value) for value in line.strip('\n').split(" ")[1:]])
    else:
        return torch.LongTensor([
            value_dict[float(value)]
            for value in line.strip('\n').split(" ")[1:]
        ])


def quantize_embed(module, nbit=32):
    """
    This function replace all embedding modules
    to QuantEmbedding layer.
    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Embedding):
            quant_embedding = QuantEmbedding(
                num_embeddings=child.num_embeddings,
                embedding_dim=child.embedding_dim,
                padding_idx=child.padding_idx,
                nbit=nbit,
                _weight=child.weight)
            # send the quant embedding layer to gpu
            # if the original embedding is on gpu
            if next(child.parameters()).is_cuda:
                quant_embedding.cuda()
            setattr(module, name, quant_embedding)
            logging.info("Replaced " + name + " in " +
                         module.__class__.__name__)
        else:
            quantize_embed(child, nbit)
    return module


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
        LongTensor, during forward for inference, the bits are extracted
        from LongTensor and put into Float32 tensor for inference.
        There are 4 ways to initialize the quantized embedding layer:
            1. a float32 tensor containing unquantized values
                _weight=<a float32 tensor>, quantized_input=False
            2. a float32 containing quantized values
                _weight=<a LongTensor>, quantized_input=True
            3. a file containing unquantized float values
                _weight=<file name>, quantized_input=False
            4. a file containing quantized float values
                _weight=<file name>, quantized_input=True
        If you use the file-style input, for reference format,
        please refer to http://nlp.stanford.edu/data/glove.6B.zip.
        """
        assert nbit in (1, 2, 4, 8, 16, 32)
        assert max_norm == None
        assert norm_type == 2.
        assert scale_grad_by_freq == False
        assert sparse == False
        if (_weight is None and embedding_file is None) or (
                _weight is not None and embedding_file is not None):
            raise Exception(
                "Should provide input either from a tensor or a file!")
        self.nbit = nbit
        # set the dimensionality of the actual compressed tensor
        if self.nbit == 32:
            self.tensor_dim = embedding_dim
        else:
            self.tensor_dim = math.ceil(embedding_dim * nbit / LONG_BITS)
        nn.Embedding.__init__(
            self,
            num_embeddings,  # we use the actual tensor dim here, otherwise will raise error
            self.tensor_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)

        # if we have an cuda input _weight, we convert it to cpu
        # so that the intermediate memory in initialization would
        # not exhaust the gpu memory.
        if _weight is not None and _weight.is_cuda:
            _weight = _weight.detach().cpu()
        if self.nbit == 32:
            # we only support forward pass
            self.weight.requires_grad = False
            if embedding_file is not None:
                _weight = self._load_from_unquant_file_to_uncompressed_tensor(
                    embedding_file)
            self.weight.copy_(_weight.data)
        else:
            self.weight = nn.Parameter(
                torch.zeros(
                    num_embeddings, self.tensor_dim, dtype=torch.int64),
                requires_grad=False)
            # record the true embeding_dim
            self.embedding_dim = embedding_dim
            # load the quantized values from file / tensor to the int64 tensor
            if self._quantized_input(_weight, embedding_file):
                if _weight is not None:
                    assert isinstance(_weight, torch.FloatTensor)
                    # the input weight is already quantized and does not need clipping/quantization
                    self._compress_tensor(_weight, do_quant=False)
                elif embedding_file is not None:
                    # the functionality of compress tensor is included in the loading function here
                    self._load_from_quant_file_to_compressed_tensor(
                        embedding_file)
            else:
                if _weight is not None:
                    assert isinstance(_weight, torch.FloatTensor)
                elif embedding_file is not None:
                    _weight = self._load_from_unquant_file_to_uncompressed_tensor(
                        embedding_file)
                # compress _weight into self.weight
                self._compress_tensor(_weight)
        logger.info("Compressed embedding to " + str(self.nbit) + " bits!")


    def _get_value_list_from_tensor(self, weight):
        # get the unique values into a list
        if isinstance(weight, torch.FloatTensor):
            weight = weight.detach().cpu().numpy()
        sorted_vals = sorted(np.unique(weight).tolist())
        return sorted_vals

    def _get_value_list_from_file(self, file_name):
        value_set = set([])
        with open(file_name, "r") as f:
            for line_id, line in enumerate(f.readlines()):
                for value in line.strip('\n').split(" ")[1:]:
                    value_set.add(float(value))
        sorted_vals = sorted(list(value_set))
        return sorted_vals

    def _quantized_input(self, weight, embedding_file):
        assert weight is None or embedding_file is None, " Can only use one out of Tensor or File as input!"
        if weight is not None:
            return len(
                self._get_value_list_from_tensor(weight)) <= 2**self.nbit
        else:
            return len(
                self._get_value_list_from_file(embedding_file)) <= 2**self.nbit

    def _load_from_unquant_file_to_uncompressed_tensor(self, file_name):
        weight = torch.zeros(self.num_embeddings, self.embedding_dim)
        # put vectors into int32 tensor
        with open(file_name, "r") as f:
            for line_id, line in enumerate(f.readlines()):
                vector = line2vec(line)
                if self.embedding_dim != vector.numel():
                    raise Exception(
                        "Dimensionality in embedding file does not match dimensionality specified for embedding layer"
                    )
                weight[line_id].copy_(vector)
        return weight

    def _compress_tensor(self, weight, do_quant=True):
        '''
        if weight is not quantized yet, we specify do_quant to quantize here
        '''
        if (weight.shape[0] != self.num_embeddings) or (weight.shape[1] !=
                                                        self.embedding_dim):
            raise Exception(
                "The shape of the input embedding does not match the compressed tensor!"
            )
        assert self.nbit != 32, "_compress_tensor should only be called when nbit < 32"
        if do_quant:
            weight, _, _ = compress.compress_uniform(
                weight.detach().cpu().numpy(),
                self.nbit,
                adaptive_range=True,
                stochastic_round=False)
        else:
            weight = weight.detach().cpu().numpy()
        # construct value dict
        sorted_vals = self._get_value_list_from_tensor(weight)
        self.register_buffer("value_list", torch.FloatTensor(sorted_vals))
        self.value_dict = {
            float(value): i
            for i, value in enumerate(sorted_vals)
        }
        assert len(sorted_vals) <= 2**self.nbit
        if len(sorted_vals) < 2**self.nbit:
            logger.warning(
                "Set of actual values is smaller than set of possible values.")
        weight = np.vectorize(self.value_dict.get)(weight)
        # compress vectors into quantized embeddings
        self.weight.copy_(
            compress_long_mat(torch.LongTensor(weight), nbit=self.nbit))

    def _load_from_quant_file_to_compressed_tensor(self, file_name):
        if self.nbit != 32:
            # construct the mapping between quantized index and quantized value
            sorted_vals = self._get_value_list_from_file(file_name)
            self.register_buffer("value_list", torch.FloatTensor(sorted_vals))
            self.value_dict = {
                float(value): i
                for i, value in enumerate(sorted_vals)
            }
            assert len(sorted_vals) <= 2**self.nbit
            if len(sorted_vals) < 2**self.nbit:
                logger.warning(
                    "Set of actual values is smaller than set of possible values."
                )
        else:
            with open(file_name, "r") as f:
                line_cnt = len(f.readlines())

        # put vectors into int64 tensor
        with open(file_name, "r") as f:
            for line_id, line in enumerate(f.readlines()):
                if self.nbit != 32:
                    vector = line2vec(line, self.value_dict)
                    self.weight[line_id].copy_(
                        compress_long_mat(vector, self.nbit))
                else:
                    self.weight[line_id].copy_(
                        torch.tensor(
                            [float(value) for value in line.split(" ")[1:]],
                            dtype=self.weight.dtype))
        if self.num_embeddings > line_id + 1:
            logger.warning(
                "The input vocab is smaller then the specified vocab size")

    def forward(self, input):
        embedding = F.embedding(input, self.weight, self.padding_idx,
                                self.max_norm, self.norm_type,
                                self.scale_grad_by_freq, self.sparse)
        if self.nbit != 32:
            embedding = decompress_long_mat(embedding, self.nbit,
                                            self.embedding_dim)
            embedding = self.value_list[embedding]
        assert self.weight.requires_grad == False, " QuantEmbedding only support fixed embedding"
        return embedding
