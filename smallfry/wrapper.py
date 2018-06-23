import numpy as np
import os
from .utils import *
import marisa_trie
import logging
import bitarray as ba
import json

class Smallfry:

    def __init__(self,path,word2idx):
        self.path = path
        self.word2idx = word2idx
        self.codebks = np.load(path+"/codebks.npy") 
        self.sfry_size = os.path.getsize(path+"/codebks.npy")
        metadata = json.loads(open(path+"/metadata",'r').read())  
        self.allots = metadata['allots']
        self.allot_indices = metadata['allot_indices']
        self.dim = metadata['dim']
        self.bin_submats = dict()     
    
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
                self.sfry_size += len(submat_ba)/8.0
                codelen = self.dim * self.allots[i]
                self.bin_submats[i] = [submat_ba[i:i+codelen] for i in range(0, len(submat_ba), codelen)]
                
    def _query(self, idx, submat_idx):
        R_i = self.allots[submat_idx] if idx >= 0 else 0
        if R_i == 0:
            return np.repeat(self.codebks[submat_idx][0],self.dim)
        offset_idx = idx - self.allot_indices[submat_idx]
        decode_d = prepare_decode_dict(self.allots[submat_idx], self.codebks[submat_idx])
        return np.array(self.bin_submats[submat_idx][offset_idx].decode(decode_d))

    def query(self, word):
        idx, submat_idx = query_prep(word, self.word2idx, self.dim, self.codebks, self.allot_indices)
        return self._query(idx, submat_idx)

    def query_idx(self, idx):
        return self._query(idx,get_submat_idx(idx, self.allot_indices))
    
    def get_size(self):
        return self.sfry_size 
