import numpy as np
import json
import bitarray as ba
import marisa_trie as mtrie
from .core import quantize
from .smallfry import Smallfry


def serialize(sfry, filepath):
    '''
    Serializes binary representation to file
    Includes metadata as {filepath}.meta
    '''
    bin_file = open(filepath, 'wb')
    meta_file = open(filepath+'.meta', 'w')

    sfry.bin_rep.tofile(bin_file)
    meta_file.write(json.dumps([sfry.dim, sfry.codebk]))

    bin_file.close()
    meta_file.close()

def deserialize(filepath):
    '''
    Loads a lloyd-max quantized binary and the metadata from file
    '''

    bin_file = open(filepath,'rb')
    meta_file = open(filepath+'.meta','r')

    bit_arr = ba.bitarray()
    bit_arr.fromfile(bin_file)

    metadata = json.loads(meta_file.read())

    return Smallfry(bit_arr, metadata[0], metadata[1])

def compress(embeddings, wordlist):
    '''
    Compresses a numpy embeddins matrix with word list
    '''
    wordtrie = mtrie.Trie(wordlist)
    sorted_embeds = np.zeros(embeddings.shape)
    for i in range(0,len(wordlist)):
        i_prime = wordtrie[wordlist[i]]
        sorted_embeds[i_prime,:] = embeddings[i,:]
    sfry = quantize(sorted_embeds)
    return sfry, wordtrie


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
    f.close()

    print("Embedding shape: " + str(embeddings.shape))
    return vocab, embeddings


