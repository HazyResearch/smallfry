import numpy as np
import argh
import os
import sys
import smallfry
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) #FOR LOCAL IMPORTS
from experimental_utils import * 
from maker.uniform_quant import stochround, midriser

demo_work_dir = ''

def test_stochround():
    for b in [1,2,4]:
        for i in range(10):
            L_prime = 1/torch.rand([1]) 
            X = torch.rand([10,10])*2*L_prime - L_prime
            X_q = torch.Tensor(stochround(X,b))
            assert len(torch.unique(X_q)) <= 2**b, "More centroids than codes!"
            quanta = [] # compute quanta
            n = 2**b - 1
            L = torch.max( torch.abs( X ))
            for i in range(n+1):
                quanta.append(i/n)
            quanta = np.array(quanta)
            quanta = (quanta*2 - 1)*L.data.numpy()
            for i in range(X_q.shape[0]):
                for j in range(X_q.shape[1]):
                    eps = 0.001 # handle numerical imprecision in transition from numpy/torch
                    assert torch.abs(X_q[i,j] - X[i,j]) < 2*L/(2**b-1), "Invalid quantization"
                    entry_is_on_point = False
                    for q in quanta:
                        if abs(X_q[i,j].data.numpy() - q) < eps:
                            entry_is_on_point = True
                    assert entry_is_on_point, "Invalid quantization"

def test_stochround_bias_simple():
    data = np.array([0.3]*10000 + [0.1]*10000)
    seeds = [1,2,3]
    bias_tol = 0.01
    for seed in seeds:
        set_seeds(seed)
        data_q = stochround(data,1)
        #data has mean of exact mean of 0.2
        bias = np.abs(np.mean(data_q) - 0.2) 
        assert bias < bias_tol, f"Stochround is potentially biased -- seed {seed} has bias of {bias}"

def test_stochround_bias_gaussian():
    bias = [-0.25,0,0.25]
    bitrates = [1,2,4]
    seeds = [1,2,3]
    bias_tol = 0.01
    for b in bias:
        for br in bitrates:
            for seed in seeds:
                set_seeds(seed)
                data = np.random.normal(b,1,1000000)
                data_q = stochround(data,br)
                bias = np.abs(np.mean(data) - np.mean(data_q)) 
                assert bias < bias_tol, f"Stochround is potentially biased -- seed {seed} has a bias of {bias}"

def test_maker():
    str_tup = ('dca','glove', '/proj/smallfry/git/smallfry/examples/data/glove.head.txt', '1234', '/proj/smallfry/embeddings', 'more_tests', '3', '8')
    os.system("python maker.py --method %s --base %s --basepath %s --seed %s --outputdir %s --rungroup %s --m %s --k %s" % str_tup)
    assert True

def test_codes_2_vec():
    m = 2
    k = 2
    d = 3
    v = 4
    codebook = np.array([[0,0,0],[1,1,1],[-2,-2,-2],[1,1,1]])
    codes = np.array([[0,0],[0,1],[1,0],[1,1]])
    codes = codes.flatten()
    dcc_mat = codes_2_vec(codes, codebook, m, k, v, d)
    dcc_mat_check = np.array([ [-2,-2,-2], [1,1,1], [-1,-1,-1], [2,2,2]  ])
    assert np.array_equal(dcc_mat, dcc_mat_check)

parser = argh.ArghParser()
parser.add_commands([test_maker, test_stochround, test_stochround_bias_simple, test_stochround_bias_gaussian])

if __name__ == '__main__':
    parser.dispatch()

