import numpy as np
import time
import os
import scipy
from scipy import sparse, linalg
from scipy.sparse import coo_matrix
from scipy.stats import norm
import math
import multiprocessing
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from cStringIO import StringIO
import struct



def rowwise_KM(row,k,max_iters=150,num_inits=3,init_dist='default'):
    kmeans = KMeans(n_clusters=k,max_iter=max_iters,n_init=num_inits,n_jobs=1).fit(row)
    return np.concatenate(kmeans.cluster_centers_[kmeans.labels_]), kmeans.labels_, kmeans.cluster_centers_
    
def bit_allocator(var_vect, bitrate, err_tol=0.1):
#computes bit allotment, needs weights as input 
    total_budget = len(var_vect)*bitrate          
    lamb_max = max(var_vect)
    lamb_min = 1e-100
    lamb = 0
    rate = 0
    while(np.abs(rate - total_budget) > err_tol):
        lamb = (lamb_max - lamb_min)/2 + lamb_min
        rate = 0.5*sum(np.log2(var_vect/np.minimum(var_vect,lamb)))
        if(rate > total_budget):
            lamb_min = lamb
        else:
            lamb_max = lamb

    return 0.5*np.log2(var_vect/np.minimum(var_vect,lamb))

def allocation_round(bit_allot_vect, sort=False):
#rounds the bit allot vect such that <1 bit goes unused in aggregate
    bit_bank = 0
    for i,a in enumerate(bit_allot_vect):
        a_up = np.ceil(a)
        a_dwn = np.floor(a)
        diffup = a_up - a
        diffdwn = a - a_dwn
        if(bit_bank > diffup):
            bit_bank -= diffup
            bit_allot_vect[i] = a_up
        else:
            bit_bank += diffdwn
            bit_allot_vect[i] = a_dwn

    bit_allot_vect = np.array(bit_allot_vect).astype(int)
    return np.array(sorted(bit_allot_vect,reverse=True)) if sort else bit_allot_vect

def text2npy(path,dim):
    embed_path = path.replace(".txt", ".npy")
    word_path = path.replace(".txt", ".word")
    word_dict_path = path.replace(".txt",".word.npy")
    f_wordout = open(word_path, "w")
    #print("convert {} to {}".format(glove_path, embed_path))
    
    words = list()
    prior = np.load("prior.npy").item()
    word2row = dict()
    p2word = dict()
	
    lines = list(open(path))
    embed_matrix = np.zeros((len(lines), dim), dtype='float32')
    p = np.zeros(len(lines), dtype='float32')
    word2idx = dict()
    
    for i, line in enumerate(lines):
        txtline = line.rstrip().split(' ')
        word = txtline[0]
        row = np.array(txtline[1:], dtype='float32')
        try:
            p[i] = prior[word]
        except KeyError: #if word not in prior... well... occurs at least ~1?
            p[i] = 1 + np.random.normal(scale=10**-5)
        word2row[word] = row
        if p[i] in p2word:
            p2word[p[i]].append(word) 
        else:
            p2word[p[i]] = list()
            p2word[p[i]].append(word)
    
    p = sorted(p, reverse=True)
    
    for i in range(0,len(p)):
        p_words = p2word[p[i]]
        for ii in range(0,len(p_words)):
            word = p_words[ii]
            vec = word2row[word]
            embed_matrix[i] = vec	
            words.append(word)
            word2idx[word] = i
            f_wordout.write(word + "\n")
  
    p = p/sum(p)
    np.save(embed_path, embed_matrix) 
    np.save(word_dict_path, word2idx)
    return embed_matrix, p, words, word2idx 
    

def mat_partition(embmat, bit_allocations):
    allots, allot_indices = np.unique(bit_allocations,return_index=True)
    submats = list()
    prev_idx = len(embmat)
    for i in range(0,len(allots)):
        cur_idx = allot_indices[i]
        submats.append(embmat[cur_idx:prev_idx])
        prev_idx = cur_idx

    return submats,allot_indices 
        

def quantize(submats):
    inf_submats = list()
    quant_submats = list()
    codebks = list()  
    for i in range(0,len(submats)):
        inf_emb, quant_emb, codebk = km_quantize(submats[i],i)
        inf_submats.append(inf_emb)
        quant_submats.append(quant_emb)
        codebks.append(codebk)
     
    inf_submats.reverse() 
    inflated_mat = np.vstack(inf_submats) 
    return inflated_mat, quant_submats, codebks 

def km_quantize(X,R):
    orig_shape = X.shape
    X = X.reshape(-1,1)
    k = 2**R
    inflated_embs, quant_embs, codebk = rowwise_KM(X,k)    
    return inflated_embs.reshape(orig_shape), quant_embs.reshape(orig_shape), codebk


def bitwrite_submats(quant_submats, codebks, path):
    sfry_path = path.replace(".txt", ".sfry")
    os.mkdir(sfry_path)

    for i in range(1,len(quant_submats)):
        s = ""
        cur_submat = quant_submats[i].reshape(-1,1)
        for ii in range(0,len(cur_submat)):
            delta_s = bin(cur_submat[ii][0])[2:]
            if len(delta_s) < i:
                delta_s = '0' * (i - len(delta_s)) + delta_s
            s += delta_s
        
        if(i > 0):
            sio = StringIO(s)
            f = open(sfry_path+"/"+str(i),'wb')
            while True:
                b = sio.read(8)
                if not b:
                    break
                if len(b) < 8:
                    b = b + '0' * (8 - len(b))
                j = int(b, 2)
                c = chr(j)
                f.write(c)
            f.close()

    return sfry_path



'''
def query(word, word2idx, allot_indices, codebk, dim):
    idx = word2idx[word]
    R_i = 0
    prev_index = 0
    submat = 0
    while R_i < len(allot_indices):
        if idx > allot_indices[R_i]:
            break
        else:
            R_i += 1

    if R_i == 0:
        return np.zeros(dim)
    
    f = open(str(R_i),'rb')
    offset_in_bits = int((idx - allot_indices[R_i])*dim*R_i)
    readend_in_bits = dim*R_i + offset_in_bits
    

    #correction is in bits from start of byte
    offset_in_bytes = offset_in_bits/8
    offset_correction = offset_in_bits%8
   
    #correction is in bits from end of byte
    readend_in_bytes = readend_in_bits/8 + 1
    readend_correction = 8-(readend_in_bits%8)

    f.seek(offset_in_bytes,0)
    row_hex = f.read(readend_in_bytes - offset_in_bytes)
    row_bitstring = ""
    for i in range(0,len(row_hex)):
        bitstring = bin(struct.unpack("B",row_hex[i])[0])[2:]
        if len(bitstring) < 8:
            bitstring = '0' * (8-len(bitstring)) + bitstring
        row_bitstring += bitstring 
    
    row_bitstring = row_bitstring[offset_correction:len(row_bitstring)-readend_correction]

    print("meep" +str(row_bitstring))
    inflated_row = np.zeros(dim)
    for i in range(0,dim):
        code = int(row_bitstring[i*R_i:(i+1)*R_i],2)
        inflated_row[i] = codebk[R_i][code]
     
    return inflated_row,row_bitstring
     
    
    
def compress(path, dim, R):
    emb_mat, p, words, word2idx = text2npy(path,dim)
    bit_allocations = allocation_round(bit_allocator(p,R),sort=True)
    submats,allot_indices = mat_partition(emb_mat, bit_allocations)
    _, quant_submats, codebks = quantize(submats)
    bitwrite_submats(quant_submats, codebks)
    return word2idx, allot_indices, codebks
'''


