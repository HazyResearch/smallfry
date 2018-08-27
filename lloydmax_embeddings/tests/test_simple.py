import numpy as np
import argh
import marisa_trie
import os
import lloydmax_embeddings as lmqe
import torch


demo_embed_path = '../../examples/data/glove.head.txt'

def torch_embed():
    X = np.random.random([100,10])
    sfry = lmqe.core.quantize(X)
    e = lmqe.embedding.SmallfryEmbedding(sfry)
    idx = torch.IntTensor([[4,1,2],[1,3,2]])
    print(e(idx))
    assert True


def io():
    vocab, embeddings = lmqe.utils.load_embeddings(demo_embed_path)
    lmq_bin, metadata, wordtrie = lmqe.utils.compress(embeddings, vocab)
    lmqe.utils.serialize(lmq_bin, metadata, 'meep')
    assert True

def compression():
    vocab, embeddings = lmqe.utils.load_embeddings(demo_embed_path)
    lmqe.utils.compress(embeddings, vocab)
    assert True


parser = argh.ArghParser()
parser.add_commands([torch_embed, io, compression])

if __name__ == '__main__':
    parser.dispatch()

