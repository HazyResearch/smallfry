import numpy as np
import argh
import os
import sys
import smallfry
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) #FOR LOCAL IMPORTS
from experimental_utils import * 
from maker.uniform_quant import *

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

def test_goldensearch_randquadratic():
    for i in range(100):
        x_min = np.random.normal()
        scale = np.random.random()
        shift = np.random.normal()
        f = lambda x : scale*(x - x_min)**2 + shift
        x_star = golden_section_search(f,x_min-10,x_min+10)
        assert np.abs(x_star - x_min) < 1e-1, f"Search procedure failure: found {x_star} with value {f(x_star)}, compared to {x_min} with value {f(x_min)}"

def test_goldensearch_randnonconvex():
    for i in range(100):
        x_min = np.random.normal()
        scale = np.random.random()
        shift = np.random.normal()
        f = lambda x : scale*np.sqrt((np.abs(x - x_min))) + shift
        x_star = golden_section_search(f,x_min-10,x_min+10)
        assert np.abs(x_star - x_min) < 1e-1, f"Search procedure failure: found {x_star} with value {f(x_star)}, compared to {x_min} with value {f(x_min)}"

def test_goldensearch_edge():
    for i in range(10):
        x_min = np.random.normal()
        scale = np.random.random()
        shift = np.random.normal()
        f = lambda x : scale*(x - x_min)**2 + shift
        x_star = golden_section_search(f,x_min=x_min,x_max=x_min+10)
        assert np.abs(x_star - x_min) < 1e-1, f"Search procedure failure: found {x_star} with value {f(x_star)}, compared to {x_min} with value {f(x_min)}"

    for i in range(10):
        x_min = np.random.normal()
        scale = np.random.random()
        shift = np.random.normal()
        f = lambda x : scale*(x - x_min)**2 + shift
        x_star = golden_section_search(f,x_min=x_min-10,x_max=x_min)
        assert np.abs(x_star - x_min) < 1e-1, f"Search procedure failure: found {x_star} with value {f(x_star)}, compared to {x_min} with value {f(x_min)}"

def test_adarange():
    data = np.arange(-1,1,0.0001)
    q1 = np.sort(np.unique(adarange(data,1)))
    assert np.abs(q1[0] + 0.5) < 0.01
    assert np.abs(q1[1] - 0.5) < 0.01
    q2 = np.sort(np.unique(adarange(data,2)))
    assert np.abs(q2[0] + 0.75) < 0.01
    assert np.abs(q2[1] + 0.25) < 0.01
    assert np.abs(q2[2] - 0.25) < 0.01
    assert np.abs(q2[3] - 0.75) < 0.01

    q1 = np.sort(np.unique(adarange(data,1)))
    assert np.abs(q1[0] + 0.5) < 0.01
    assert np.abs(q1[1] - 0.5) < 0.01
    q2 = np.sort(np.unique(adarange(data,2)))
    assert np.abs(q2[0] + 0.75) < 0.01
    assert np.abs(q2[1] + 0.25) < 0.01
    assert np.abs(q2[2] - 0.25) < 0.01
    assert np.abs(q2[3] - 0.75) < 0.01

def test_naiverange():
    data = np.arange(-1,1,0.0001)
    q1 = np.sort(np.unique(naiveuni(data,1)))
    assert (q1[0] + 1) < 0.01
    assert (q1[1] - 1) < 0.01
    q2 = np.sort(np.unique(naiveuni(data,2)))
    assert (q2[0] + 1) < 0.01
    assert (q2[1] + 1/3) < 0.01
    assert (q2[2] - 1/3) < 0.01
    assert (q2[3] - 1) < 0.01

    data = 2*(np.random.random(10000)-0.5)
    q1 = np.sort(np.unique(naiveuni(data,1)))
    assert (q1[0] + 1) < 0.01
    assert (q1[1] - 1) < 0.01
    q2 = np.sort(np.unique(naiveuni(data,2)))
    assert (q2[0] + 1) < 0.01
    assert (q2[1] + 1/3) < 0.01
    assert (q2[2] - 1/3) < 0.01
    assert (q2[3] - 1) < 0.01

def test_clipnoquant():
    eps = 0.01
    data = 2*(np.random.random(10000)-0.5)
    X1 = clip_no_quant(data,1)
    X2 = clip_no_quant(data,2)
    assert len(np.unique(X1)) > 2 and len(np.unique(X2)) > 4
    for d in X1:
        assert d > -0.5-eps and d < 0.5+eps 
    for d in X2:
        assert d > -0.75-eps and d < 0.75+eps

def test_affine_transform():
    data = np.array([-1,1])
    for L in [10*np.random.random() for i in range(100)]:
        for b in [1,2,3,4,5]:
            affine_data = affine_transform(L*data,L,b)
            assert np.isclose(affine_data, [0,2**b-1], 0.01).all() 
            data_origin = affine_transform(affine_data,L,b,invert=True)
            assert np.isclose(data_origin, [-L,L], 0.01).all() 

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
parser.add_commands([test_maker, test_stochround, test_stochround_bias_simple, test_stochround_bias_gaussian, test_naiveuni_bias_gaussian])

if __name__ == '__main__':
    parser.dispatch()

