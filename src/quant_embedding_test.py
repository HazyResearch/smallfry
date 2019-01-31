from quant_embedding import compress_long_vec
from quant_embedding import decompress_long_vec
from quant_embedding import QuantEmbedding
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
        compressed = compress_long_vec(input, n_bit)
        decompressed = decompress_long_vec(compressed, n_bit, dim=n_dim)
        # print(input)
        # print(decompressed)
        assert torch.all(torch.eq(input, decompressed))

    def generate_embedding_file(self,
                                n_bit,
                                n_dim,
                                n_word,
                                file_name=EMBEDDING_TEST_FILE):
        n_val = int(2**n_bit)
        value_list = np.random.rand(n_val)
        index = np.random.randint(low=0, high=n_val, size=(n_word, n_dim))
        embedding = value_list[index]
        with open(file_name, "w") as f:
            for i in range(n_word):
                line = "x" + str(i) + "".join(
                    [" " + str(val) for val in embedding[i]]) + "\n"
                f.write(line)
            logger.info("generated embedding, nbit, ndim, n_word " +
                        str(n_bit) + " / " + str(n_dim) + "/ " + str(n_word))
        return torch.FloatTensor(embedding)

    def get_embeddings_for_test(self, embedding_file=EMBEDDING_TEST_FILE):
        n_dim = int(np.random.randint(low=1, high=100))
        n_bit = int(np.random.choice([2, 4, 8, 16]))
        n_word = np.random.choice(np.arange(1, 100))
        ref_embedding = self.generate_embedding_file(
            n_dim=n_dim, n_bit=n_bit, n_word=n_word, file_name=embedding_file)

        # test if the file is loaded correctly
        # test 32 bit representation
        embedding = QuantEmbedding(
            num_embeddings=n_word,
            embedding_dim=n_dim,
            padding_idx=0,
            nbit=32,
            embedding_file=embedding_file)
        # test non 32 bit representation
        quant_embedding = QuantEmbedding(
            num_embeddings=n_word,
            embedding_dim=n_dim,
            padding_idx=0,
            nbit=n_bit,
            embedding_file=embedding_file)
        assert embedding.embedding_dim == ref_embedding.size(-1)
        assert quant_embedding.embedding_dim == ref_embedding.size(-1)
        return ref_embedding, embedding, quant_embedding

    def forward(self, cuda=False):
        ref_embedding, embedding, quant_embedding = self.get_embeddings_for_test(
            embedding_file=EMBEDDING_TEST_FILE)
        n_dim = embedding.weight.size(-1)
        n_word = int(embedding.weight.size(0))
        batch_size = np.random.randint(low=3, high=16)
        length = np.random.randint(low=10, high=50)
        input = torch.LongTensor(batch_size, length, n_dim).random_(to=n_word)
        if cuda:
            ref_embedding = ref_embedding.cuda()
            embedding = embedding.cuda()
            quant_embedding = quant_embedding.cuda()
            input = input.cuda()
        ref_out = ref_embedding[input]
        out = embedding(input)
        quant_out = quant_embedding(input)
        assert torch.all(torch.eq(ref_out, out))
        assert torch.all(torch.eq(out, quant_out))

    def test_forward_cpu(self):
        self.forward(cuda=False)

    def test_forward_gpu(self):
        self.forward(cuda=True)

if __name__ == "__main__":
    unittest.main()