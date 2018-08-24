import numpy as np
import json
import bitarray as ba
import marisa_trie as mtrie
from .core import quantize

def serialize(bit_arr, metadata, out_path):
    '''
    Writes a lloyd-max quantized binary and metadata to a directory
    '''
    lmqbin_filepath = out_path + '.lmqbin'
    metadata_filepath = out_path + '.meta'

    lmqbin_file = open(lmqbin_filepath,'wb')
    metadata_file = open(metadata_filepath,'w')

    bit_arr.tofile(lmqbin_file)
    metadata_file.write(json.dumps(metadata))

def deserialize(lmqbin_filepath, metadata_filepath):
    '''
    Loads a lloyd-max quantized binary and the metadata from file
    '''

    lmqbin_file = open(lmqbin_filepath,'rb')
    metadata_file = open(metadata_filepath,'r')

    bit_arr = ba.bitarray()
    bit_arr.fromfile(lmqbin_file)

    return bit_arr, json.loads(metadata_file.read())

def compress(embeddings, wordlist):
    '''
    Compresses a numpy embeddins matrix with word list
    '''
    wordtrie = mtrie.Trie(wordlist)
    sorted_embeds = np.zeros(np.shape)
    for i in range(0,len(wordlist)):
        i_prime = wordtrie[wordlist[i]]
        sorted_embeds[i_prime,:] = embeddings[i,:]
    lmq_bin, metadata = quantize(sorted_embeds)
    return lmq_bin, metadata, wordtrie


def load_embeddings(embeds_txt_filepath):
    """
    Loads a GloVe embedding at 'filename'. Returns a vector of strings that 
    represents the vocabulary and a 2-D numpy matrix that is the embeddings. 
    """
    f = open(embeds_txt_filepath, "r") 
    lines = f.readlines()
    vocab = []
    embeddings = []
    for line in lines:
        values = line.strip("\n").split(" ")
        vocab.append(values.pop(0))
        embeddings.append([float(v) for v in values])
    embeddings = np.array(embeddings)
    vocab = np.array(vocab)
    f.close()

    print("Embedding shape: " + str(embedding.shape))
    return vocab, embedding


