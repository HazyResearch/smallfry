import smallfry as sfry
import tests
import numpy as np
import os
import pytest

def gen_uniform_embs(uniprior=False):
    nums = np.array([i for i in range(0,1000)])
    uni_test_prior = dict()
    for i in range(0,1000):
        uni_test_prior[str(i)] = 10*np.random.random() if not uniprior else 1 
    np.savetxt("wordnums",nums,fmt='%i')
    np.save("uni_test_prior",uni_test_prior)
    unimat = 2*np.random.random([1000,100])-1
    np.savetxt("uni.mat",unimat,fmt='%.12f')
    os.system("paste -d ' '  wordnums uni.mat > uniembs.txt")
    word2idx, sfry_path = sfry.compress("uniembs.txt", "uni_test_prior.npy", write_inflated=True, word_rep="dict")
    return word2idx, sfry_path


def test_inflation_fidelity():
    word2idx, sfry_path = gen_uniform_embs()
    #TODO fix inflated naming!!!
    check_mmap = tests.check_inflation("uniembs.txt.inflated_None", sfry_path, word2idx, mmap=True)
    check_file = tests.check_inflation("uniembs.txt.inflated_None", sfry_path, word2idx, mmap=False)
   
    assert(check_mmap and check_file)

def test_approx_codebk_correctness():
    word2idx, sfry_path = gen_uniform_embs()
    codebks = np.load(sfry_path+"/codebks.npy")
    


    assert(a == b)

def test_weighted_fronorm():

#TODO tests: 1) check bitrate usage, 2) 
    

test_inflation_fidelity()
