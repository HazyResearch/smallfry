import smallfry as sfry
import tests
import numpy as np
import os
import pytest

def gen_uniform_embs():
    nums = np.array([i for i in range(0,1000)])
    uni_test_prior = dict()
    for i in range(0,1000):
        uni_test_prior[str(i)] = 10*np.random.random()
    np.savetxt("wordnums",nums,fmt='%i')
    np.save("uni_test_prior",uni_test_prior)
    unimat = 2*np.random.random([1000,50])-1
    np.savetxt("uni.mat",unimat,fmt='%.12f')
    os.system("paste -d ' ' uni.mat wordnums > uniembs.txt")
    word2idx, sfry_path = sfry.compress("uniembs.txt", "uni_test_prior.npy", write_inflated=True, word_rep="dict")
    return word2idx, sfry_path


def test_inflation_fidelity():
    word2dix, sfry_path = gen_uniform_embs()
    
    
     
    assert(a == b)

def test_good():
    a = 1
    b = 2
    assert(a == b)


