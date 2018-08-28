import numpy as np
import argh
import marisa_trie
import os
import smallfry as lmqe
import torch


demo_embed_path = '../examples/data/glove.head.txt'

def test_torch_embed():
    X = np.random.random([100,10])
    sfry = lmqe.smallfry.Smallfry.quantize(X)
    e = lmqe.embedding.SmallfryEmbedding(sfry)
    idx = torch.IntTensor([[4,1,2],[1,3,2]])
    print(e(idx))
    assert True


parser = argh.ArghParser()
parser.add_commands([test_torch_embed])

if __name__ == '__main__':
    parser.dispatch()

