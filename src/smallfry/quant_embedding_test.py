from quant_embedding import compress_long_mat
from quant_embedding import decompress_long_mat
from quant_embedding import QuantEmbedding
from quant_embedding import quantize_embed
import compress
from unittest import TestCase
import torch
import numpy as np
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("quant embedding test")

EMBEDDING_TEST_FILE = "./test_embed.txt"


class QuantEmbeddingTest(TestCase):
    def test_compress_decompress_funcs(self):
        # test the compress and decompress functions are reverse to each other
        n_bit = int(np.random.choice([2, 4, 8, 16, 32]))
        n_vals = int(2**n_bit)
        n_dim = np.random.randint(low=2, high=100)
        batch_size = np.random.randint(low=3, high=16)
        length = np.random.randint(low=10, high=50)
        input = torch.LongTensor(batch_size, length, n_dim).random_(to=n_vals)
        compressed = compress_long_mat(input, n_bit)
        decompressed = decompress_long_mat(compressed, n_bit, dim=n_dim)
        # print(input)
        # print(decompressed)
        assert torch.all(torch.eq(input, decompressed))

    def test_embeding_replacement_func(self):
        layer1 = torch.nn.Embedding(100, 10)
        layer2 = torch.nn.Embedding(200, 20)
        layer3 = torch.nn.Embedding(300, 30)
        layer4 = torch.nn.Embedding(400, 40)
        module_list1 = torch.nn.ModuleList([layer1, layer2])
        module_list2 = torch.nn.ModuleList([layer3, layer4])
        module_list = torch.nn.ModuleList([module_list1, module_list2])
        module_list_comp = quantize_embed(module_list, nbit=4)
        assert isinstance(module_list_comp[0][0], QuantEmbedding)
        assert isinstance(module_list_comp[0][1], QuantEmbedding)
        assert isinstance(module_list_comp[1][0], QuantEmbedding)
        assert isinstance(module_list_comp[1][1], QuantEmbedding)

        layer1 = torch.nn.Embedding(100, 10)
        layer2 = torch.nn.Embedding(200, 20)
        layer3 = torch.nn.Embedding(300, 30)
        layer4 = torch.nn.Embedding(400, 40)
        module_list1 = torch.nn.Sequential(layer1, layer2)
        module_list2 = torch.nn.Sequential(layer3, layer4)
        module_list = torch.nn.Sequential(module_list1, module_list2)
        module_list_comp = quantize_embed(module_list, nbit=4)
        assert isinstance(module_list_comp[0][0], QuantEmbedding)
        assert isinstance(module_list_comp[0][1], QuantEmbedding)
        assert isinstance(module_list_comp[1][0], QuantEmbedding)
        assert isinstance(module_list_comp[1][1], QuantEmbedding)

    def generate_embedding_file(self,
                                n_bit,
                                n_dim,
                                n_word,
                                quantized_input,
                                file_name=EMBEDDING_TEST_FILE):
        if quantized_input:
            n_val = int(2**n_bit)
            value_list = np.random.rand(n_val)
            index = np.random.randint(low=0, high=n_val, size=(n_word, n_dim))
            embedding = value_list[index]
        else:
            embedding = np.random.rand(n_word, n_dim)
        with open(file_name, "w") as f:
            for i in range(n_word):
                line = "x" + str(i) + "".join(
                    [" " + str(val) for val in embedding[i]]) + "\n"
                f.write(line)
            logger.info("generated embedding, nbit, ndim, n_word " +
                        str(n_bit) + " / " + str(n_dim) + "/ " + str(n_word))
        return torch.FloatTensor(embedding)

    def get_embeddings_for_test(self, quantized_input=False, use_file=True):
        n_dim = int(np.random.randint(low=1, high=100))
        n_bit = int(np.random.choice([2, 4, 8, 16]))
        n_word = np.random.choice(np.arange(1, 100))

        input_embedding = self.generate_embedding_file(
            n_dim=n_dim,
            n_bit=n_bit,
            n_word=n_word,
            quantized_input=quantized_input,
            file_name=EMBEDDING_TEST_FILE)
        if use_file:
            weight = None
            embedding_file = EMBEDDING_TEST_FILE
        else:
            weight = input_embedding.clone()
            embedding_file = None

        # test if the file is loaded correctly
        # test 32 bit representation
        embedding = QuantEmbedding(
            num_embeddings=n_word,
            embedding_dim=n_dim,
            padding_idx=0,
            nbit=32,
            _weight=weight,
            embedding_file=embedding_file)

        # test non 32 bit representation
        quant_embedding = QuantEmbedding(
            num_embeddings=n_word,
            embedding_dim=n_dim,
            padding_idx=0,
            nbit=n_bit,
            _weight=weight,
            embedding_file=embedding_file)

        if quantized_input or np.unique(input_embedding).size <= 2**n_bit:
            ref_embedding = input_embedding.clone()
        else:
            # we only quantize when there is not enough bits
            ref_embedding, _, _ = compress.compress_uniform(
                input_embedding.cpu().numpy(),
                n_bit,
                adaptive_range=True,
                stochastic_round=False)
            ref_embedding = torch.FloatTensor(ref_embedding)

        assert embedding.embedding_dim == ref_embedding.size(-1)
        assert quant_embedding.embedding_dim == ref_embedding.size(-1)
        return input_embedding, ref_embedding, embedding, quant_embedding

    def forward(self, cuda=False):
        config_list = [("load quantized file as input test ", {
            "quantized_input": True,
            "use_file": True
        }), ("load unquantized file as input test ", {
            "quantized_input": False,
            "use_file": True
        }), ("load unquantized tensor as input test ", {
            "quantized_input": False,
            "use_file": False
        }), ("load quantized tensor as input test ", {
            "quantized_input": True,
            "use_file": False
        })]
        for info, config in config_list:
            input_embedding, ref_embedding, embedding, quant_embedding = \
                self.get_embeddings_for_test(**config)
            # ref_embedding, embedding, quant_embedding = self.get_embeddings_for_test(
            #     embedding_file=EMBEDDING_TEST_FILE)
            n_dim = embedding.weight.size(-1)
            n_word = int(embedding.weight.size(0))
            batch_size = np.random.randint(low=3, high=16)
            length = np.random.randint(low=10, high=50)
            input = torch.LongTensor(batch_size, length,
                                     n_dim).random_(to=n_word)
            if cuda:
                input_embedding = input_embedding.cuda()
                ref_embedding = ref_embedding.cuda()
                embedding = embedding.cuda()
                quant_embedding = quant_embedding.cuda()
                input = input.cuda()
            assert quant_embedding.weight.is_cuda == cuda
            assert quant_embedding.value_list.is_cuda == cuda

            input_out = input_embedding[input]
            ref_out = ref_embedding[input]
            out = embedding(input)
            quant_out = quant_embedding(input)
            assert torch.all(torch.eq(input_out, out))
            assert torch.all(torch.eq(ref_out, quant_out))
            logger.info(info + "passed!")

    def test_forward_cpu(self):
        self.forward(cuda=False)

    def test_forward_gpu(self):
        self.forward(cuda=True)


if __name__ == "__main__":
    unittest.main()