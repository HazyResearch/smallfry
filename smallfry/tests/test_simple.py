import numpy as np
import smallfry as sfry
import argh
import marisa_trie
import os


def parse_txtemb(path):
    npy_mat = np.loadtxt(os.popen("cat "+str(path)+" | cut -d ' '  -f2- "))
    npy_wordlist = list(filter(None, os.popen("awk '{print $1}' "+str(path)).read().split('\n')))
    embs = dict()
    
    i = 0
    for w in npy_wordlist:
        embs[w] = npy_mat[i]
        i += 1
    return embs


def read_emb(path, fmt):
    if fmt == 'trie':
        embs = marisa_trie.Trie()
        return embs.load(str(path))

    elif fmt == 'dict':
        return np.load(str(path)).item()

    elif fmt == 'inflated':
        return parse_txtemb(str(path))

    return None

def read_embs(source, compressed, comp_fmt): #TODO replace this with above
    src_npy = np.loadtxt(os.popen("cat "+str(source)+" | cut -d ' '  -f2- "))
    src_wordlist = os.popen("awk '{print $1}' "+str(source)).read().split('\n')
    src_embs = parse_txtemb(str(source))
      
    comp_embs = None

    if comp_fmt == 'trie':
        comp_embs = marisa_trie.Trie()
        comp_embs.load(str(compressed))    

    elif comp_fmt == 'dict':
        comp_embs = np.load(str(compressed)).item()

    elif comp_fmt == 'inflated':
        comp_embs = parse_txtemb(str(compressed))
    
    return src_embs, comp_embs

def prior_vocab_union(prior,vocab):
    for v in vocab:
        if not v in prior:
            prior[v] = 1
    
    return prior

def compute_square_w_fronorm(source, compressed, priorpath):
    prior = np.load(str(priorpath), encoding='latin1').item()
   
    prior = prior_vocab_union(prior, source.keys())
    #TODO: this prior only works for 1s 
 
    return sum([prior[w]*np.linalg.norm(source[w] - compressed[w])**2 for w in source])


def check_inflation(inflated_path, sfry_path, word2idx, mmap=True):
    inflated_embs = read_emb(str(inflated_path), fmt='inflated')
    c = 0
    check_passed = True 
    if mmap:
        my_sfry = sfry.load(str(sfry_path), word2idx)
        for w in word2idx:
            c += 1
            if c % 10000 == 0: 
                print(c)
            if np.linalg.norm(my_sfry.query(w) - inflated_embs[w]) > 0.01:
                print(my_sfry.query(w))
                print(inflated_embs[w])
                print("Error on word "+w)
                check_passed = False
                break   
    else:
        for w in word2idx:
            c += 1
            if c % 10000 == 0: 
                print(c)
            if np.linalg.norm(sfry.query(w, word2idx, sfry_path) - inflated_embs[w]) > 0.01:
                print(sfry.query(w, word2idx, sfry_path))
                print(inflated_embs[w])
                
                print("Error on word "+w)
                check_passed = False
                break   

    print("done")
    return check_passed 


def weighted_fronorm(src_path, compressed_path, priorpath, comp_fmt='inflated'):
    src_embs, comp_embs = read_embs(src_path, compressed_path, comp_fmt)
    return compute_square_w_fronorm(src_embs, comp_embs, priorpath)
    

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
    word2idx, sfry_path = sfry.compress("uniembs.txt", "uni_test_prior.npy", "out" , write_inflated=True, word_rep="dict")
    return word2idx, sfry_path


def test_inflation_fidelity():
    word2idx, sfry_path = gen_uniform_embs()
    #TODO fix inflated naming!!!
    infpath = sfry_path+"/sfry.inflated"
    check_mmap = check_inflation(infpath, sfry_path, word2idx, mmap=True)
    #check_file = check_inflation(infpath, sfry_path, word2idx, mmap=False)
   
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
    infpath = sfry_path+"/sfry.inflated" 
    #we can solve this problem explicitly to get ~0.041 as the per entry distortion
    net_dist = 0.0833*100*1000
    dist = weighted_fronorm("uniembs.txt",infpath,"uni_test_prior.npy")
    

    assert(np.abs(net_dist -dist) < 100)

parser = argh.ArghParser()
parser.add_commands([weighted_fronorm, check_inflation])

if __name__ == '__main__':
    parser.dispatch()

