import numpy as np
import os
import scipy
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from io import StringIO
import struct
import logging
import sys
import shutil
import bitarray as ba
import marisa_trie
#SMALL-FRY CORE: Direct usage not recommended

#TODO: bad method
def rowwise_KM_minibatch(row, k, max_iters=90, num_inits=3):#TODO hardcoding + method call
    kmeans = MiniBatchKMeans(n_clusters=k,max_iter=max_iters).fit(row)
    return np.concatenate(kmeans.cluster_centers_[kmeans.labels_]), kmeans.labels_, kmeans.cluster_centers_
    
#TODO bad method
def rowwise_KM(row,k,max_iters=120,num_inits=3,init_dist='default'): #TODO hardcoding
#uses sklearn scalar KMeans which implements Lloyd's iterative algo
    kmeans = KMeans(n_clusters=k,max_iter=max_iters,n_init=num_inits,n_jobs=3).fit(row)
    return np.concatenate(kmeans.cluster_centers_[kmeans.labels_]), kmeans.labels_, kmeans.cluster_centers_
    

def bit_allocator(weight_vect, bitrate, err_tol=0.1):
#computes bit allotment, needs weights as input 
    total_budget = len(weight_vect)*bitrate          
    lamb_max = max(weight_vect)
    lamb_min = 1e-100 #magic number but is probably ok?
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

#I think this method is fine?
def allocation_round(bit_allot_vect, sort=True):
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

#this really should be in the dev branch
def downsample(bit_allot_vect, dim, max_bitrate, topdwn_upsamp=True):
#This method downsamples such that the k-means is always assigns less cluster than points
    budget = 0
    V = len(bit_allot_vect)
    maxrate = np.floor(np.log2(dim)) if max_bitrate == None else int(max_bitrate)
    for i in range(0,V):
        if bit_allot_vect[i] > maxrate:
            budget += bit_allot_vect[i] - maxrate
            bit_allot_vect[i] = maxrate 
    
    while(budget > 0):
        prev_budget = budget
        for i in range(0,V):
            j = i if topdwn_upsamp else V-i-1
            if bit_allot_vect[j] < maxrate and budget > 0:
                bit_allot_vect[i] += 1
                budget -=1
        if prev_budget == budget:
            break 

    return sorted(bit_allot_vect,reverse=True)

#bad method I cant recall why I have the dtypes, or why I have some magic number scales
def text2npy(inpath, outpath, priorpath, word_rep):
#preprocessing textfile embeddings input
    words = list()
    prior = np.load(priorpath,encoding='latin1').item()
    word2row = dict()
    p2word = dict()
	
    lines = list(open(inpath))
    p = np.zeros(len(lines), dtype='float32')
    word2idx = dict()
    
    dim = 0
    for i, line in enumerate(lines):
        txtline = line.rstrip().split(' ')
        word = txtline[0]
        row = np.array(txtline[1:], dtype='float32')
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
        
        if i % 10000 == 0: 
            logging.debug("Parsing txt... on line # "+str(i)+" out of "+str(len(lines)))
             
    p = sorted(p, reverse=True)
    p_unique = sorted(np.unique(p), reverse=True)
    embed_matrix = np.zeros((len(lines), dim), dtype='float32')
    
    logging.debug("Embeddings parse complete, preparing word representation...")

    iii = 0
    wordout = []
    for i, priors in enumerate(p_unique):
        p_words = p2word[priors]
        for ii in range(0, len(p_words)):
            word = p_words[ii]
            vec = word2row[word]
            embed_matrix[iii] = vec
            words.append(word)	
            word2idx[word] = iii
            iii += 1
    
    p = np.array(p)/sum(p)

    assert word_rep in ['dict', 'trie', 'list'], "bad word rep"

    if word_rep == 'dict': 
        np.save(outpath+"/words.dict", word2idx)
        #logging.debug("Word dictionary written to "+word_dict_path)
    elif word_rep == 'trie': 
        keys = list(word2idx.keys())
        vals = list([[word2idx[k]] for k in keys])
        word_trie = marisa_trie.RecordTrie('I',zip(keys,vals))
        word_trie.save(outpath+"/words.trie.marisa")
        #logging.debug("Word trie written to "+word_trie_path)
    elif word_rep == 'list':
        f = open(outpath+"/words.list.txt",'w')
        f.write("\n".join(words))
        f.close()
    
    return embed_matrix, p, words, word2idx, dim 
    
#probably the worst one in here... yeah needs work
def npy2text(npy_mat,words,writepath):
    f = open(writepath+str(".words"),'w')
    f.write("\n".join(words))
    f.close()
    np.savetxt(writepath+".mat",npy_mat,fmt='%.12f')
    os.system("paste -d ' ' "+writepath+str(".words")+" "+writepath+str(".mat")+" > "+writepath)

#this one seems ok
def mat_partition(embmat, bit_allocations):
#partitions the matrix in submatrices based on the bit allocations
    allots, allot_indices = np.unique(bit_allocations,return_index=True)
    submats = list()
    prev_idx = len(embmat)
    for i in range(0,len(allots)):
        cur_idx = allot_indices[i]
        submats.append(embmat[cur_idx:prev_idx])
        prev_idx = cur_idx

    logging.debug("Partitioning into "+str(len(submats))+" submatrices...")
    return submats, allots, allot_indices 

#the 0.05 should be only in dev branch... but otherwise, is ok?
def matpart_adjuster(submats, allots, allot_indices, V, max_partition=0.05):
    adjusted_submats = list() 
    adjusted_allots = list()
    adjusted_allot_indices = list()
    old_allot_indices = list(allot_indices)
    old_allot_indices = [V] + old_allot_indices
    max_part_size = int(V*max_partition)  
    old_allot_indices = list(reversed(old_allot_indices))
     
    for i in range(0,len(old_allot_indices)-1):
        j = len(submats)-i-1 
        a_idx_end = old_allot_indices[i+1]
        a_idx_strt = old_allot_indices[i]
        part_size = a_idx_end - a_idx_strt
        num_subparts = int(part_size / max_part_size)
        if num_subparts <= 1:
            adjusted_submats.append(submats[j])
            adjusted_allots.append(allots[j])
            adjusted_allot_indices.append(allot_indices[j])
        else:
            for ii in range(0,num_subparts):
                offset = ii*max_part_size
                this_part_size = max_part_size if ii < num_subparts-1 else V
                adjusted_submats.append(submats[j][offset:offset+this_part_size])
                adjusted_allots.append(allots[j]) 
                adjusted_allot_indices.append(a_idx_strt+offset)

    return adjusted_submats, adjusted_allots, adjusted_allot_indices
        
#seems ok?
def quantize(submats, allots, minibatch):
#quantizes each submatrix
    inf_submats = list()
    quant_submats = list()
    codebks = list() 
    for i in range(0,len(submats)):
        logging.debug("Quantizing submat # "+str(i)+"...")
        inf_emb, quant_emb, codebk = km_quantize(submats[i], allots[i], minibatch)
        inf_submats.append(inf_emb)
        quant_submats.append(quant_emb)
        codebks.append(codebk)
     
    inflated_mat = np.vstack(inf_submats) 
    return inflated_mat, quant_submats, codebks
 
#seems ok?
def km_quantize(X, R, minibatch):
#k-means for submatrix
    orig_shape = X.shape
    X = X.reshape(-1,1)
    k = 1 << R
    KM = rowwise_KM_minibatch if minibatch else rowwise_KM
    inflated_embs, quant_embs, codebk = KM(X,k)  
    return inflated_embs.reshape(orig_shape), quant_embs.reshape(orig_shape), codebk

#seems oK   ?
def prepare_encode_dict(bitrate):
    nVals = 1 << bitrate
    assert bitrate <= 9, "TODO" #other than this 
    codelst = [ba.bitarray(bin(i)[2:].zfill(bitrate)) for i in range(0,nVals)]
    return dict(enumerate(codelst))
        

def prepare_decode_dict(nbits, codebook):
    decode_d = dict()
    nVals    = 1 << nbits #TODO_CMR is this to ensure interger? Instead of 2**nbits? Will copy from now on
    assert nbits <= 9, "Maximum bit size of 8 bits?" #depends
    the_bits = np.unpackbits(np.array([[i] for i in range(nVals)], dtype=np.uint8), axis=1)[:,8-nbits:8] 
    for i in range(nVals):
        bi = ba.bitarray(list(the_bits[i,:]))
        decode_d[codebook[i][0]] = bi
    return decode_d      

#seems ok?
def bitwrite_submats(quant_submats, codebks, allots, path):
    for i in range(0,len(quant_submats)):
        logging.debug("Generating bitwise representation for submatrix # "+str(i)+"...") 
        cur_submat = [i[0] for i in quant_submats[i].reshape(-1,1)]
        submat_ba = ba.bitarray()
        submat_ba.encode(prepare_encode_dict(allots[i]),cur_submat)
        submat_ba.tofile(open(path+"/submat"+str(i),'wb'))

#ok?
def get_submat_idx(idx, allot_indices):
    a_i = 0
    while a_i < len(allot_indices)-1:
        if idx <  allot_indices[a_i+1]: break
        else: a_i += 1
    return a_i
  
#TODO LIST
def get_word_idx(word2idx, word):
    word_rep_type = type(word2idx)
    supported_types = [dict, marisa_trie.RecordTrie, list]
    assert word_rep_type in supported_types, "Word representation object is not of a supported type"

    if word in word2idx:
        if type(word2idx) is dict:
            return word2idx[word]
        if type(word2idx) is marisa_trie.RecordTrie:
            return word2idx[word][0][0]
        if type(word2idx) is list:#TODO
            return exit()
    else: return -1

#seems ok
def query_prep(word, word2idx, dim, codebks,allot_indices):
        idx = get_word_idx(word2idx, word)
        submat_idx = -1
        if idx > -1:
            submat_idx = get_submat_idx(idx, allot_indices)
        return idx, submat_idx

