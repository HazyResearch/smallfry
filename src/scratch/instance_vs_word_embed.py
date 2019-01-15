import numpy as np
import sys
sys.path.append("./")
from utils import load_embeddings
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('')


def get_spectrum(X):
	X  = (X + X.T)/2.0
	np.testing.assert_array_equal(X, X.T)
	sigma, V = np.linalg.eigh(X)
	return sigma

def get_weighted_embedding(embedding, word_dict, freq_dict):
	freq_list = [freq for word, freq in freq_dict.items()]
	for word, i in word_dict.items():
		# if word == '<unk>':
		# 	print(embedding[i])
		# print(i, freq_dict[word])
		if word == '<unk>':
			embedding[i] *= min(freq_list)
			logger.info("treat unk with lowest freq " + str(min(freq_list)))
		else:
			embedding[i] *= np.sqrt(float(freq_dict[word]))
		# print(word, freq_dict[word])
	return embedding


def get_word_dict(word_list):
	word_dict = {}
	for i, word in enumerate(word_list):
		word_dict[word] = i
	return word_dict


def load_word_freq(path):
	freq_dict = {}
	with open(path, 'r', encoding='utf8') as f:
		for line in f.readlines():
			row = line.strip('\n').split(" ")
			freq_dict[row[0]] = float(row[1])
	return freq_dict


if __name__ == "__main__":
	if sys.argv[1] == "compute":
		embed_file = "/dfs/scratch0/avnermay/smallfry/base_embeddings/glove-wiki400k-am/rungroup,2018-12-18-trainGlove_embedtype,glove_corpus,wiki400k_embeddim,400_threads,72_embeds.txt"
		freq_file = "/dfs/scratch0/avnermay/smallfry/corpora/wiki400k/vocab_wiki400k.txt"

		embedding, word_list = load_embeddings(embed_file)
		logger.info("embedding loaded")
		word_freq = load_word_freq(freq_file)
		logger.info("frequence loaded")
		word_dict = get_word_dict(word_list)
		instance_embed = get_weighted_embedding(embedding.copy(), word_dict, word_freq)
		logger.info("instance embedding generated")
		# n_word = float(embedding.shape[0])

		word_spectrum = get_spectrum(np.dot(embedding.T, embedding))
		instance_spectrum = get_spectrum(np.dot(instance_embed.T, instance_embed))

		word_spec_file = "word_spec.npy"
		instance_spec_file = "instance_spec.npy"
		np.save(word_spec_file, word_spectrum)
		np.save(instance_spec_file, instance_spectrum)

	elif sys.argv[1] == "plot":
		import matplotlib.pyplot as plt
		word_spec_file = "word_spec.npy"
		instance_spec_file = "instance_spec.npy"
		word_spectrum = np.load(word_spec_file)
		instance_spectrum = np.load(instance_spec_file)

		plt.figure()
		plt.semilogy(word_spectrum)
		plt.savefig(word_spec_file.replace(".npy", ".pdf"))
		plt.close()

		plt.figure()
		plt.semilogy(instance_spectrum)
		plt.savefig(instance_spec_file.replace(".npy", ".pdf"))
		plt.close()





