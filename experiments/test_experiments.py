import numpy as np
import argh
import os
import smallfry
import torch
from codes_2_vec import codes_2_vec

demo_work_dir = ''

def test_dca():
    pass

def test_maker():
    #os.system("python maker.py --method %s --dca %s --basepath %s --seed %s --outputdir %s --rungroup %s --m %s --k %s" % ())
    pass

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
parser.add_commands([test_dca])

if __name__ == '__main__':
    parser.dispatch()

