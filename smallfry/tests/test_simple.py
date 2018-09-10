import numpy as np
import argh
import os
import smallfry
import torch


demo_embed_path = '../examples/data/glove.head.txt'

def test_query():
    X = np.random.random([100,10])
    sfry = smallfry.smallfry.Smallfry.quantize(X)
    idx = [(np.random.random(3)*100).astype(int)]
    res = sfry(torch.IntTensor([idx,idx]))
    assert torch.all(torch.eq(res[0], res[1]))

def test_io():
    X = np.random.random([100,10])
    sfry = smallfry.smallfry.Smallfry.quantize(X)
    idx = [(np.random.random(3)*100).astype(int)]
    smallfry.smallfry.Smallfry.serialize(sfry, 'test_io.sfry')
    sfry_deserial = smallfry.smallfry.Smallfry.deserialize('test_io.sfry')
    assert sfry.decode(np.array(idx)).all() ==  sfry_deserial.decode(np.array(idx)).all()

def test_kmeans():
    X = np.random.random([1000,10])
    sfry = smallfry.smallfry.Smallfry.quantize(X)
    passing = True
    for i in range(1000):
        X_hat = sfry.decode(np.array(i))
        for j in range(10):
            if abs(X[i,j] - X_hat[j]) > 0.27: 
                passing = False
    assert passing

def test_blocklen():
    X = np.random.random([100,10])
    sfry = smallfry.smallfry.Smallfry.quantize(X,b=1,block_len=2)
    idx = [(np.random.random(3)*100).astype(int)]
    res = sfry(torch.IntTensor([idx,idx]))
    assert torch.all(torch.eq(res[0], res[1]))


parser = argh.ArghParser()
parser.add_commands([test_query, test_io, test_kmeans])

if __name__ == '__main__':
    parser.dispatch()

