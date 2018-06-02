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
from io import StringIO
import struct
import logging

#INTERNAL UTILS FOR SMALL-FRY API: Direct usage not recommended

#global variables here
#these are automatically cleaned when needed, if only the API methods are called
DP_memo = dict()
args_table = dict() 
lin_prefix_sums = np.array([])
quad_prefix_sums = np.array([])
clusters = 0

def centroid_factory(i,j):
#TODO: this method is under active development -- not recommended for use
    global lin_prefix_sums
    global quad_prefix_sums
    linsum = lin_prefix_sums[j] - lin_prefix_sums[i-1]
    quadsum = quad_prefix_sums[j] - quad_prefix_sums[i-1]
    centroid = 1.0*linsum/(j-i+1)
    return (j-i+1)*centroid**2 - 2*centroid * linsum + quadsum
    
def DP_solver(memokey):
#TODO: this method is under active development -- not recommended for use
    global DP_memo
    if memokey in DP_memo:
        return DP_memo[memokey]
    
    sol = 0
    argsol = memokey[1]
    if memokey[1] <= memokey[0]:
        sol = 0
    elif memokey[0] == 1:
        sol = centroid_factory(1,memokey[1])
    else:
        DP = [DP_solver((memokey[0]-1,jj-1)) + centroid_factory(jj,memokey[1])\
            for jj in range(1,memokey[1]+1)]
        sol =  min(DP)
        argsol = np.argmin(DP)+1
        
    DP_memo[memokey] = sol
    args_table[memokey] = argsol
    return sol
        

def monotone_matrix_query(m,j):
#TODO: this method is under active development -- not recommended for use    
    global clusters
    cost = 0
    memokey = (clusters-1,min(j-1,m))
    if j <= m:
        cost = centroid_factory(j,m)
    return -1*(DP_solver(memokey) + cost)
    
def fast_KM(row,k):
#TODO: this method is under active development -- not recommended for use
    vector = np.array(sorted(row))
    global clusters
    global lin_prefix_sums  
    global quad_prefix_sums
    global DP_memo
    global args_table
    DP_memo = dict()
    args_table = dict()
    clusters = k
    lin_prefix_sums = np.zeros(len(vector)+1)
    quad_prefix_sums = np.zeros(len(vector)+1)

    for i in range(0,len(vector)+1):
        lin_prefix_sums[i] = sum(vector[0:i])
        quad_prefix_sums[i] = sum(vector[0:i]**2) 
    
    rows = [i for i in range(1,len(vector)+1)]
    cols = [m for m in range(1,len(vector)+1)]
    lookup = monotone_matrix_query
   
    solution = monotone_mat_search(rows,cols,lookup)
    clusterstarts = np.zeros(k,dtype=int)
    codebk = np.zeros(k)
    clusterstarts[k-1] = solution[len(vector)]
    clusterstarts[0] = 1
    for i in range(1,k-1):
        l = k-i
        clusterstarts[l-1] = args_table[(l,clusterstarts[l]-1)] 
       
    for i in range(0,k):
        if i+1 < len(clusterstarts):
            a = clusterstarts[i+1]-1
        else:
            a = len(vector)
        b = clusterstarts[i]-1
        linsum = lin_prefix_sums[a] - lin_prefix_sums[b]
        codebk[i] = 1.0*linsum/(a-b)
 
    inflated_embs = np.zeros(len(vector))
    quant_embs = np.zeros(len(vector),dtype=int)
    s = 0

    for i in range(0,len(vector)):
        quant_embs[i] = np.argmin(np.abs(row[i] - codebk))
        inflated_embs[i] = codebk[quant_embs[i]]
     
    codelist = list()
    for i in range(0,len(codebk)):
        codelist.append(np.array([codebk[i]]))

    codelist = np.array(codelist)
    

    return inflated_embs,quant_embs,codelist

def monotone_mat_search(rows,cols,lookup):
#TODO: this method is under active development -- not recommended for use
    xrange = range
    # base case of recursion
    if not rows: return {}
    # reduce phase: make number of columns at most equal to number of rows
    stack = []
    for c in cols:
        while len(stack) >= 1 and \
          lookup(rows[len(stack)-1],stack[-1]) < lookup(rows[len(stack)-1],c):
            stack.pop()
        if len(stack) != len(rows):
            stack.append(c)

    cols = stack
    # recursive call to search for every odd row
    result = monotone_mat_search([rows[i] for i in xrange(1,len(rows),2)],cols,lookup)
    # go back and fill in the even rows
    c = 0
    for r in xrange(0,len(rows),2):
        row = rows[r]
        if r == len(rows) - 1:
            cc = len(cols)-1  # if r is last row, search through last col
        else:
            cc = c            # otherwise only until pos of max in row r+1
            target = result[rows[r+1]]
            while cols[cc] != target:
                cc += 1
        result[row] = max([ (lookup(row,cols[x]),-x,cols[x]) \
                            for x in xrange(c,cc+1) ]) [2]
        c = cc
    return result
    

def rowwise_KM(row,k,max_iters=200,num_inits=10,init_dist='default'):
#uses sklearn scalar KMeans which implements Lloyd's iterative algo
    kmeans = KMeans(n_clusters=k,max_iter=max_iters,n_init=num_inits,n_jobs=1).fit(row)
    return np.concatenate(kmeans.cluster_centers_[kmeans.labels_]), kmeans.labels_, kmeans.cluster_centers_
    
def bit_allocator(weight_vect, bitrate, err_tol=0.1):
#computes bit allotment, needs weights as input 
    total_budget = len(weight_vect)*bitrate          
    lamb_max = max(weight_vect)
    lamb_min = 1e-100
    lamb = 0
    rate = 0
    while(np.abs(rate - total_budget) > err_tol):
        lamb = (lamb_max - lamb_min)/2 + lamb_min
        rate = 0.5*sum(np.log2(weight_vect/np.minimum(weight_vect,lamb)))
        if(rate > total_budget):
            lamb_min = lamb
        else:
            lamb_max = lamb

    return 0.5*np.log2(weight_vect/np.minimum(weight_vect,lamb))

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


def bit_allocator_2D(spectrum, weights, bits_per_entry):
#TODO: this method is under active development -- not recommended for use
    num_cols = len(spectrum)
    num_rows = len(weights)
    total_budget = bits_per_entry*num_cols*num_rows
    lamb = 2.0**(-4)
    while(compute_bit_allot_grid(spectrum,weights,num_cols,num_rows,lamb)[0] < total_budget):
        lamb = lamb*2

    upper_b = np.ceil(lamb)
    lower_b = np.floor(lamb/2)
    while(upper_b - lower_b > 1.1):
        mid_p = 0.5*(upper_b + lower_b)
        budget_usage,_ = compute_bit_allot_grid(spectrum, weights, num_cols, num_rows, mid_p)
        increase = (budget_usage < total_budget)
        mid_p = np.floor(mid_p) if increase else np.ceil(mid_p)
        if increase:
            lower_b = mid_p
        else:
            upper_b = mid_p

    return compute_bit_allot_grid(spectrum, weights, num_cols, num_rows, lower_b)
         

def compute_bit_allot_grid(spectrum, weights, num_cols, num_rows, lamb):
#TODO: this method is under active development -- not recommended for use
    bit_allot_grid = 0
    with np.errstate(divide='ignore'):
        bit_allot_grid = np.add.outer( np.floor(np.log2(spectrum)) , np.floor(0.5*np.log2(weights)) + lamb )
    bit_allot_grid = np.maximum(bit_allot_grid, np.zeros([num_cols,num_rows]))
    budget_usage = np.sum(bit_allot_grid)
    return budget_usage,bit_allot_grid.astype(int)
    


def downsample(bit_allot_vect,dim):
#This method downsamples such that the k-means is always assigns less cluster than points
    budget = 0
    V = len(bit_allot_vect)
    maxrate = np.floor(np.log2(dim))
    for i in range(0,V):
        if bit_allot_vect[i] > maxrate:
            budget += bit_allot_vect[i] - maxrate
            bit_allot_vect[i] = maxrate
        elif bit_allot_vect[i] < maxrate and budget > 0:
            bit_allot_vect[i] += 1
            budget -=1

    print(bit_allot_vect)
    return bit_allot_vect
      

def text2npy(path,priorpath, word_rep,dim):
#preprocssing textfile embeddings input
    embed_path = path+".npy"
    word_path = path+".word"
    word_dict_path = path+".word.npy"
    word_trie_path = path+".word.marisa-trie"
    f_wordout = open(word_path, "w")
    
    words = list()
    prior = np.load(priorpath,encoding='latin1').item()
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
        if dim == None:
            dim = len(row)
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
            if word_rep == 'list':
                f_wordout.write(word + "\n")
  
    p = p/sum(p)
    #np.save(embed_path, embed_matrix) 
    if word_rep == 'dict'
        np.save(word_dict_path, word2idx)
    if word_rep == 'trie':
        import marisa_trie
        keys = list(word2idx.keys())
        vals = [word2idx[k] for k in keys]
        word_trie = marisa_trie.RecordTrie('<H',zip(keys,vals))
        np.save(word_trie_path, word_trie)
    
    return embed_matrix, p, words, word2idx 
    

def mat_partition(embmat, bit_allocations):
#partitions the matrix in submatrices based on the bit allocations
    allots, allot_indices = np.unique(bit_allocations,return_index=True)
    submats = list()
    prev_idx = len(embmat)
    for i in range(0,len(allots)):
        cur_idx = allot_indices[i]
        submats.append(embmat[cur_idx:prev_idx])
        prev_idx = cur_idx

    return submats,allot_indices 
        

def quantize(submats):
#quantizes each submatrix
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
#k-means for submatrix
    orig_shape = X.shape
    X = X.reshape(-1,1)
    k = 2**R
    inflated_embs, quant_embs, codebk = rowwise_KM(X,k)    
    return inflated_embs.reshape(orig_shape), quant_embs.reshape(orig_shape), codebk


def bitwrite_submats(quant_submats, codebks, path, endian = 'little'):
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
                f.write(int.to_bytes(j,1,endian))
            f.close()

    return sfry_path


