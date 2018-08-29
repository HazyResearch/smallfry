import numpy as np
import argh
import marisa_trie
import os
import smallfry.smallfry
import torch


demo_embed_path = '../examples/data/glove.head.txt'

def test_query():
    X = np.random.random([100,10])
    sfry = smallfry.smallfry.Smallfry.quantize(X)
    idx = [(np.random.random(3)*100).astype(int)]
    res = sfry(torch.IntTensor([idx,idx]))
    print(res)
    assert torch.all(torch.eq(res[0], res[1]))

def test_io():
    pass

def test_kmeans():
    pass

parser = argh.ArghParser()
parser.add_commands([test_query, test_io, test_kmeans])

if __name__ == '__main__':
    parser.dispatch()

