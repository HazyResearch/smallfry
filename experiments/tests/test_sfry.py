import numpy as np
import argh
import os
import smallfry
import torch
import sys

demo_embed_path = 'examples/data/glove.head.txt'

def test_query():
    X = np.random.random([100,10])
    sfry = smallfry.smallfry.Smallfry.quantize(X)
    idx = [(np.random.random(3)*100).astype(int)]
    res = sfry(torch.IntTensor([idx,idx]))
    x = res[0].data.numpy()
    y = res[1].data.numpy()
    assert np.array_equal(x,y)

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

def test_dynprog():
    X = np.random.random([1000,10])
    sfry = smallfry.smallfry.Smallfry.quantize(X,solver='dynprog')
    passing = True
    for i in range(1000):
        X_hat = sfry.decode(np.array(i))
        for j in range(10):
            if abs(X[i,j] - X_hat[j]) > 0.27: 
                passing = False
    assert passing

def test_blocklen12():
    X = np.random.random([100,10])
    sfry = smallfry.smallfry.Smallfry.quantize(X,b=1,block_len=2)
    idx = [(np.random.random(3)*100).astype(int)]
    res = sfry(torch.IntTensor([idx,idx]))
    x = res[0].data.numpy()
    y = res[1].data.numpy()
    assert np.array_equal(x,y)

def test_blocklen22():
    X = np.random.random([100,10])
    sfry = smallfry.smallfry.Smallfry.quantize(X,b=2,block_len=2)
    idx = [(np.random.random(3)*100).astype(int)]
    res = sfry(torch.IntTensor([idx,idx]))
    x = res[0].data.numpy()
    y = res[1].data.numpy()
    assert np.array_equal(x,y)

def test_blocklen25():
    X = np.random.random([100,10])
    sfry = smallfry.smallfry.Smallfry.quantize(X,b=2,block_len=5)
    idx = [(np.random.random(3)*100).astype(int)]
    res = sfry(torch.IntTensor([idx,idx]))
    x = res[0].data.numpy()
    y = res[1].data.numpy()
    assert np.array_equal(x,y)

def test_vector():
    Y1 = [1,2,3,4,5,6,7,8]
    Y2 = [9,10,11,12,13,14,15,16]
    X = np.array([Y1,Y2])
    sfry = smallfry.smallfry.Smallfry.quantize(X,b=3,block_len=2)
    index_tensor = np.array([0,1])
    rows = sfry.decode(index_tensor)
    # note this lambda only works if list elements are unique
    list_eq = lambda l1,l2 : len(set(l1).intersection(l2)) == len(l1)
    assert list_eq(Y1,rows[0]) and list_eq(Y2,rows[1])

def test_zeros():
    X = np.zeros([10,10])
    sfry = smallfry.smallfry.Smallfry.quantize(X)
    idx = np.array(list(range(10)))
    list_eq = lambda l1,l2 : len(set(l1).intersection(l2)) == len(l1)
    assert np.array_equal( sfry.decode(idx).flatten(), X.flatten() ) 

def test_blocklen_many():
    block_lens = [1,2,3,7,8,11,15,20]
    for bl in block_lens:
        for br in [1,4,8]:
            X = np.random.random([100,10*bl])
            sfry = smallfry.smallfry.Smallfry.quantize(X,b=br,block_len=bl)
            idx = [(np.random.random(3)*100).astype(int)]
            res = sfry(torch.IntTensor([idx,idx]))
            x = res[0].data.numpy()
            y = res[1].data.numpy()
            assert np.array_equal(x,y)

def test_determinstic_easy():
    Y1 = [0,1,2]
    Y2 = [9,10,11]
    X = np.array([Y1,Y2])
    sfry = smallfry.smallfry.Smallfry.quantize(X,b=1,block_len=1)
    index_tensor = np.array([0,1])
    rows = sfry.decode(index_tensor) 
    assert np.array_equal(rows[0], np.array([1,1,1])) and np.array_equal(rows[1], np.array([10,10,10]))

def test_load_embeddings():
    X,v = smallfry.utils.load_embeddings(demo_embed_path)
    assert X.shape == (1000,50) and len(v) == 1000 and v[0] == 'the'

def test_dec_frobenius():
    X,v = smallfry.utils.load_embeddings(demo_embed_path)
    for bl in [1,2,5]:
        fros = [np.Infinity]
        for br in [1,2,4]:
            sfry = smallfry.smallfry.Smallfry.quantize(X,b=br, block_len=bl)
            X_q = sfry.decode(np.array(list(range(1000))))
            fros.append(np.linalg.norm(X-X_q))
            assert fros[-1] < fros[-2]


parser = argh.ArghParser()
parser.add_commands([test_query, test_io, test_kmeans, test_zeros, test_dynprog])

if __name__ == '__main__':
    parser.dispatch()

