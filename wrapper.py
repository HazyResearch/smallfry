import numpy as np
import os
import utils
import marisa_trie
import logging

class Smallfry:
    word2idx = dict()
    path = ""
    memmap_reps = dict()
    codebks = np.zeros(1)
    allot_indices = np.zeros(1)

    def __init__(self,path,word2idx):
        self.path = path
        self.word2idx = word2idx
        self.codebks = np.load(path+"/codebks.npy")
        self.allot_indices = np.load(path+"/metadata.npy") 
        self.dim = np.load(path+"/dim.npy")  
        
        directory = os.fsencode(path)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.startswith("submat"): 
                i = filename[-1]
                fullpath = path+'/'+filename
                f_size = os.path.getsize(fullpath)
                self.memmap_reps[int(i)] = np.memmap(fullpath, dtype='uint8', mode='readonly', shape=(f_size))
                
    def query(self, word):
        idx, R_i, OofV = utils.query_prep(word, self.word2idx, self.dim, self.codebks, self.allot_indices)
        if R_i == 0:
            return OofV
        offset, readend, offset_correction, readend_correction = utils.get_scan_params(idx,self.allot_indices,R_i,self.dim)
        return utils.query_exec(self.memmap_reps[R_i][offset:readend], offset_correction, readend_correction, R_i, self.codebks, self.dim)
