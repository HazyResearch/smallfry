import numpy as np
import os
from .utils import *
import marisa_trie
import logging

class Smallfry:
    word2idx = dict()
    path = ""
    memmap_reps = dict()
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
                i = filename[-1]
                fullpath = path+'/'+filename
                f_size = os.path.getsize(fullpath)
                self.memmap_reps[int(i)] = np.memmap(fullpath, dtype='uint8', mode='readonly', shape=(f_size))
                
    def query(self, word):
        idx, submat_idx, OofV = query_prep(word, self.word2idx, self.dim, self.codebks, self.allot_indices)
        R_i = self.allots[submat_idx]
        if R_i == 0:
            return OofV
        print(submat_idx)
        print(R_i)
        offset, readend, offset_correction, readend_correction = get_scan_params(idx,self.allot_indices,R_i, submat_idx, self.dim)
        print(offset)
        return query_exec(self.memmap_reps[R_i][offset:readend], offset_correction, readend_correction, R_i, submat_idx, self.codebks, self.dim)
