import numpy as np
import os
from .utils import *
import marisa_trie
import logging
import bitarray as ba

class Smallfry:
    #TODO_CMR: On the RARE occasion that I make a python class, I like putting member vars this way above the init such that I know what "type" they are suppose to be... happy to ax this if it is poor form... I  made this up myself years ago. LMK
    word2idx = dict()
    path = ""
    bin_submats = dict()
    codebks = np.zeros(1)
    allots = np.zeros(1)
    allot_indices = np.zeros(1)
    dim = 0

    def __init__(self,path,word2idx):
        self.path = path
        self.word2idx = word2idx
        self.codebks = np.load(path+"/codebks.npy")
        self.allots = np.load(path+"/metadata/ballocs.npy")
        self.allot_indices = np.load(path+"/metadata/ballocs_idx.npy")
        self.dim = np.load(path+"/metadata/dim.npy")  
        
        directory = os.fsencode(path)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.startswith("submat"): 
                i = int(filename.replace("submat",""))
                if self.allots[i] == 0: continue
                fullpath = path+'/'+filename
                f_size = os.path.getsize(fullpath)
                submat_ba = ba.bitarray()
                submat_ba.fromfile(open(fullpath,'rb'))
                codelen = self.dim * self.allots[i]
                self.bin_submats[i] = [submat_ba[i:i+codelen] for i in range(0, len(submat_ba), codelen)]
                
    def _query(self, idx, submat_idx):
        R_i = self.allots[submat_idx] if idx >= 0 else 0
        if R_i == 0:
            return np.repeat(self.codebks[submat_idx][0],self.dim)
        offset_idx = idx - self.allot_indices[submat_idx]
        decode_d = prepare_decode_dict(self.allots[submat_idx], self.codebks[submat_idx])
        return self.bin_submats[submat_idx][offset_idx].decode(decode_d)

    def query(self, word):
        idx, submat_idx = query_prep(word, self.word2idx, self.dim, self.codebks, self.allot_indices)
        return self._query(idx, submat_idx)

    def query_idx(self, idx):
        return self._query(idx,get_submat_idx(idx, self.allot_indices))
