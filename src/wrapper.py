import numpy as np
import os
import utils
import marisa_trie


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
                

    def get_word_idx(self, word):
        if type(self.word2idx) is dict:
            return self.word2idx[word]
        elif type(self.word2idx) is marisa_trie.RecordTrie:
            return self.word2idx[word][0][0]
        else:
            #throw error
            return None


    def query(self, word):
        idx = self.get_word_idx(word)
        R_i = utils.get_submat_idx(idx, self.allot_indices)
        if R_i == 0:
            return np.repeat(self.codebks[0][0],self.dim)
        offset, readend, offset_correction, readend_correction = utils.get_scan_params(idx,self.allot_indices,R_i,self.dim)
        
        rowbytes = self.memmap_reps[R_i][offset:readend]
        bitstring = utils.parse_row(rowbytes, offset_correction, readend_correction)
        return utils.decode_row(bitstring, R_i, self.codebks, self.dim) 
