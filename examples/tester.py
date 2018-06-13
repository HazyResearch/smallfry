import numpy as np
import smallfry as sfry
import argh
import marisa_trie
import os


def parse_txtemb(path):
    npy_mat = np.loadtxt(os.popen("cat "+str(path)+" | cut -d ' '  -f2- "))
    npy_wordlist = os.popen("awk '{print $1}' "+str(path)).read().split('\n')
    embs = dict()
    
    i = 0
    for w in npy_wordlist:
        if w == '':
            continue
        embs[w] = npy_mat[i]
        i += 1
    
    return embs


def read_emb(path, fmt):
    if fmt == 'trie':
        embs = marisa_trie.Trie()
        return embs.load(str(path))

    elif fmt == 'dict':
        return np.load(str(compressed)).item()

    elif fmt == 'inflated':
        return parse_txtemb(str(compressed))

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
    
    return sum([prior[w]*np.linalg.norm(source[w] - compressed[w]) for w in source])/len(source)
################
################
def check_inflation(inflated_path, sfry_path, word2idx_path, mmap=True):#TODO support disk query
    inflated_embs = read_emb(str(inflated_path), fmt='inflated'))
    word2idx = np.load(word2idx_path).item()
    query = None
    if mmap:
        my_sfry = sfry.load(str(sfry_path), word2idx)
        for w in word2idx:
            if np.linalg.norm(my_sfry.query(w) - inflated_embs[w]) > 0.01:
                print("Error on word "+w)
                break

    return 


def test_weighted_fronorm(src_path, compressed_path, priorpath, comp_fmt='inflated'):
    src_embs, comp_embs = read_embs(src_path, compressed_path, comp_fmt)
    return compute_square_w_fronorm(src_embs, comp_embs, priorpath)
    
parser = argh.ArghParser()
parser.add_commands([test_weighted_fronorm])

if __name__ == '__main__':
    parser.dispatch() 
