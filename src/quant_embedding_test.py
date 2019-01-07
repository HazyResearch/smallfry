from quant_embedding import compress_long_vec
from quant_embedding import decompress_long_vec
from quant_embedding import QuantEmbedding
from unittest import TestCase
import torch
import numpy as np

class QuantEmbeddingTest(TestCase):
	def test_compress_decompress_funcs(self):
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


	def test_loading_file(self):
		pass

	def test_forward(self):
		pass


if __name__ == "__main__":
    unittest.main()