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
    infpath = "uniembs.txt.sfry.inflated_R_1_seed_None_max_bitrate_None_batch_full_sampling_True"
    check_mmap = tests.check_inflation(infpath, sfry_path, word2idx, mmap=True)
    check_file = tests.check_inflation(infpath, sfry_path, word2idx, mmap=False)
   
    assert(check_mmap and check_file)

def test_approx_codebk_correctness():
    word2idx, sfry_path = gen_uniform_embs()
    codebks = np.load(sfry_path+"/codebks.npy")
    codes_seem_approx_cor = True
    for codebk in codebks:
        if len(codebk) == 4:
            srtbk = sorted(codebk)
            if np.abs(srtbk[0] + 0.75) > 0.05:
                codes_seem_approx_cor = False
            elif np.abs(srtbk[1] + 0.25) > 0.05:
                codes_seem_approx_cor = False
            elif np.abs(srtbk[2] - 0.25) > 0.05:
                codes_seem_approx_cor = False
            elif np.abs(srtbk[3] - 0.75) > 0.05:
                codes_seem_approx_cor = False

        elif len(codebk) == 2:
            srtbk = sorted(codebk)
            if np.abs(srtbk[0] + 0.5) > 0.05:
                codes_seem_approx_cor = False
            elif np.abs(srtbk[1] - 0.5) > 0.05:
                codes_seem_approx_cor = False
           
        elif len(codebk) == 1:
            if np.abs(codebk[0]) > 0.05:
                codes_seem_approx_cor = False 
    
    assert(codes_seem_approx_cor)

def test_weighted_fronorm():
    word2idx, sfry_path = gen_uniform_embs(uniprior=True)
    infpath = "uniembs.txt.sfry.inflated_R_1_seed_None_max_bitrate_None_batch_full_sampling_True" 
    #we can solve this problem explicitly to get ~0.041 as the per entry distortion
    net_dist = 0.0833*100*1000
    dist = tests.test_weighted_fronorm("uniembs.txt",infpath,"uni_test_prior.npy")
    

    assert(np.abs(net_dist -dist) < 100)

    
test_weighted_fronorm()
test_approx_codebk_correctness()
test_inflation_fidelity()
