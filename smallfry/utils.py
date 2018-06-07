import numpy as np
import os
import scipy
from sklearn.cluster import KMeans
from io import StringIO
import struct
import logging
import sys

#INTERNAL UTILS FOR SMALL-FRY API: Direct usage not recommended

#global variables here
#these are automatically cleaned when needed, if only the API methods are called
DP_memo = dict()
args_table = dict() 
lin_prefix_sums = np.array([])
quad_prefix_sums = np.array([])
clusters = 0


def rowwise_KM(row,k,max_iters=120,num_inits=5,init_dist='default'):
#uses sklearn scalar KMeans which implements Lloyd's iterative algo
    kmeans = KMeans(n_clusters=k,max_iter=max_iters,n_init=num_inits,n_jobs=5).fit(row)
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


def downsample(bit_allot_vect, dim, topdwn_upsamp=True):
#This method downsamples such that the k-means is always assigns less cluster than points
    budget = 0
    V = len(bit_allot_vect)
    maxrate = np.floor(np.log2(dim))
    for i in range(0,V):
        if bit_allot_vect[i] > maxrate:
            budget += bit_allot_vect[i] - maxrate
            bit_allot_vect[i] = maxrate 
    
    while(budget > 0):
        prev_budget = budget
        for i in range(0,V):
            j = i if topdwn_upsamp else V-i-1
            print(bit_allot_vect[j])
            if bit_allot_vect[j] < maxrate and budget > 0:
                bit_allot_vect[i] += 1
                budget -=1
        if prev_budget == budget:
            break 

    print(bit_allot_vect)

    return sorted(bit_allot_vect,reverse=True)
      

def text2npy(inpath, outpath, priorpath, word_rep, write_rep):
#preprocessing textfile embeddings input
    embed_path = outpath+".npy"
    word_path = outpath+".word"
    word_dict_path = outpath+".word.npy"
    word_trie_path = outpath+".word.marisa"
    if not write_rep:
        f_wordout = open(word_path, "w")
    
    words = list()
    prior = np.load(priorpath,encoding='latin1').item()
    word2row = dict()
    p2word = dict()
	
    lines = list(open(inpath))
    p = np.zeros(len(lines), dtype='float32')
    word2idx = dict()

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
    embed_matrix = np.zeros((len(lines), dim), dtype='float32')
    
    logging.debug("Embeddings parse complete, preparing word representation...")
   
    for i in range(0,len(p)):
        p_words = p2word[p[i]]
        for ii in range(0,len(p_words)):
            word = p_words[ii]
            vec = word2row[word]
            embed_matrix[i] = vec	
            words.append(word)
            word2idx[word] = i
            if not word_rep:
                f_wordout.write(word + "\n")
  
    p = p/sum(p)
    if write_rep:
        if word_rep == 'dict': 
            np.save(word_dict_path, word2idx)
            logging.debug("Word dictionary written to "+word_dict_path)
        if word_rep == 'trie': 
            import marisa_trie
            keys = list(word2idx.keys())
            vals = list([[word2idx[k]] for k in keys])
            word_trie = marisa_trie.RecordTrie('I',zip(keys,vals))
            word_trie.save(word_trie_path)
            logging.debug("Word trie written to "+word_trie_path)

    return embed_matrix, p, words, word2idx, dim 
    

def npy2text(npy_mat,words,writepath):
    f = open(writepath,'w')
    for i,w in enumerate(words):
        f.write(w)
        for j in range(0,len(npy_mat[i])):
            f.write(" "+str(npy_mat[i][j]))
        f.write('\n')

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
        
def matpart_adjuster(submats, allots, allot_indices, V, max_partition=0.05):
    adjusted_submats = list() 
    adjusted_allots = list()
    adjusted_allot_indices = list()
    old_allot_indices = list(allot_indices)
    old_allot_indices = [V] + old_allot_indices
    max_part_size = int(V*0.05)    
    
   
    for i in range(0,len(old_allot_indices)-1):
        a_idx_end = old_allot_indices[i]-1
        a_idx_strt = old_allot_indices[i+1]
        part_size = a_idx_end - a_idx_strt
        num_subparts = int(part_size / max_part_size)
        if num_subparts <= 1:
            adjusted_submats.append(submats[i])
            adjusted_allots.append(allots[i])
            adjusted_allot_indices.append(allot_indices[i])
        else:
            for ii in range(0,num_subparts):
                offset = ii*max_part_size
                this_part_size = max_part_size if ii < num_subparts-1 else V
                adjusted_submats.append(submats[i][offset:offset+this_part_size])
                adjusted_allots.append(allots[i]) 
                adjusted_allot_indices.append(a_idx_strt+offset
)
    adjusted_allot_indices = sorted(adjusted_allot_indices, reverse=True)
    adjusted_allots = sorted(adjusted_allots)
    return adjusted_submats, adjusted_allots, adjusted_allot_indices
        

def quantize(submats, allots):
#quantizes each submatrix
    inf_submats = list()
    quant_submats = list()
    codebks = list()  
    for i in range(0,len(submats)):
        logging.debug("Quantizing submat # "+str(i)+"...")
        inf_emb, quant_emb, codebk = km_quantize(submats[i], allots[i])
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


def bitwrite_submats(quant_submats, codebks, allots, path):
    sfry_path = path+".sfry"
    os.mkdir(sfry_path)

    for i in range(1,len(quant_submats)):
        s = ""
        logging.debug("Generating bitwise representation for submatrix # "+str(i)+"...") 
        cur_submat = quant_submats[i].reshape(-1,1)
        for ii in range(0,len(cur_submat)):
            delta_s = bin(cur_submat[ii][0])[2:]
            if len(delta_s) < allots[i]:
                delta_s = '0' * (allots[i]- len(delta_s)) + delta_s
            s += delta_s
       
        if(allots[i] > 0):
            sio = StringIO(s)
            f = open(sfry_path+"/"+"submat"+str(i),'wb')
            while True:
                b = sio.read(8)
                if not b:
                    break
                if len(b) < 8:
                    b = b + '0' * (8 - len(b))
                j = int(b, 2)
                f.write(int.to_bytes(j,1,sys.byteorder))
            f.close()

    return sfry_path


def get_submat_idx(idx, allot_indices):
    R_i = 0
    prev_index = 0
    while R_i < len(allot_indices):
        if idx >= allot_indices[R_i]:
            break
        else:
            R_i += 1

    return R_i
  

def get_scan_params(idx, allot_indices, R_i, dim):
    offset_in_bits = int((idx - allot_indices[R_i])*dim*R_i)
    readend_in_bits = dim*R_i + offset_in_bits
    
    #correction is in bits from start of byte
    offset_in_bytes = offset_in_bits/8
    offset_correction = offset_in_bits%8
   
    #correction is in bits from end of byte
    readend_in_bytes = readend_in_bits/8 + 1
    readend_correction = 8-(readend_in_bits%8) - 0**(readend_in_bits%8)*8

    offset_in_bytes = int(offset_in_bytes)
    readend_in_bytes = int(readend_in_bytes)
    offset_correction = int(offset_correction)
    readend_correction = int(readend_correction)
    return offset_in_bytes, readend_in_bytes, offset_correction, readend_correction


def parse_row(rowbytes, offset_correction, readend_correction):
    row_bitstring = ""
    for i in range(0,len(rowbytes)):
        bitstring = bin(rowbytes[i])[2:]   
    #bitstring = bin(struct.unpack("B",rowbytes[i])[0])[2:]
        if len(bitstring) < 8:
            bitstring = '0' * (8-len(bitstring)) + bitstring
        row_bitstring += bitstring 
    
    row_bitstring = row_bitstring[offset_correction:len(row_bitstring)-readend_correction]
    return row_bitstring


def decode_row(row_bitstring, R_i, codebks, dim):

    inflated_row = np.zeros(dim)
    for i in range(0,dim):
        code = int(row_bitstring[i*R_i:(i+1)*R_i],2)
        inflated_row[i] = codebks[R_i][code]
     
    return inflated_row


def get_word_idx(word, word2idx):
    if word in word2idx: #TODO check this
        if type(word2idx) is dict:
            return word2idx[word]
        elif type(word2idx) is marisa_trie.RecordTrie:
            return word2idx[word][0][0]
        else:
            logging.debug("Requested word is out-of-vocabulary...")
            return -1
    else:
        logging.warn("Improper word representation provided... using out-of-vocabulary code")
        return -1


def query_prep(word, word2idx, dim, codebks, allot_indices):
        idx = get_word_idx(word, word2idx)
        if idx == -1:
            R_i = 0
        else:
            R_i = get_submat_idx(idx, allot_indices)
        OofV = np.repeat(codebks[0][0], dim)
        return idx, R_i, OofV

        
def query_exec(rowbytes, offset_correction, readend_correction, R_i, codebks, dim):
    bitstring = parse_row(rowbytes, offset_correction, readend_correction)
    return decode_row(bitstring, R_i, codebks, dim)


def usr_idx_prep(word, uid):
    word2idx = dict()
    word2idx[word] = uid
    return word2idx


##########FUNCTIONS BELOW THIS LINE UNDER ACTIVE DEVELOPMENT -- DO NOT USE

def centroid_factory(i,j):
#TODO: this method is under active development -- do not use
    global lin_prefix_sums
    global quad_prefix_sums
    linsum = lin_prefix_sums[j] - lin_prefix_sums[i-1]
    quadsum = quad_prefix_sums[j] - quad_prefix_sums[i-1]
    centroid = 1.0*linsum/(j-i+1)
    return (j-i+1)*centroid**2 - 2*centroid * linsum + quadsum
    
def DP_solver(memokey):
#TODO: this method is under active development -- do not use
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
#TODO: this method is under active development -- do not use    
    global clusters
    cost = 0
    memokey = (clusters-1,min(j-1,m))
    if j <= m:
        cost = centroid_factory(j,m)
    return -1*(DP_solver(memokey) + cost)
    
def fast_KM(row,k):
#TODO: this method is under active development -- do not use
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
#TODO: this method is under active development -- do not use
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
 

def bit_allocator_2D(spectrum, weights, bits_per_entry):
#TODO: this method is under active development -- do not use
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
#TODO: this method is under active development -- do not use
    bit_allot_grid = 0
    with np.errstate(divide='ignore'):
        bit_allot_grid = np.add.outer( np.floor(np.log2(spectrum)) , np.floor(0.5*np.log2(weights)) + lamb )
    bit_allot_grid = np.maximum(bit_allot_grid, np.zeros([num_cols,num_rows]))
    budget_usage = np.sum(bit_allot_grid)
    return budget_usage,bit_allot_grid.astype(int)
    

   
