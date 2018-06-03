import utils
import wrapper
import struct
import numpy as np
import argh
import logging
#TODO:
#make dimensions in metadata
#autodetect dimensions
#load thing into tmpfile in memory
#Trie for words

#These three methods can be used either as command lines utils or a programmatic API


def load(sfry_path, word2idx):
    return wrapper.Smallfry(sfry_path, word2idx) 

def query2(word, word2idx, sfry_path):
    
    dim = np.load(sfry_path+"/dim.npy")
    codebks = np.load(sfry_path+"/codebks.npy")
    allot_indices = np.load(sfry_path+"/metadata.npy")
    idx, R_i, OofV = query_prep(word, word2idx, dim, codebks, allot_indices)
    if R_i == 0:    
        return OofV
    offset, readend, offset_correction, readend_correction = utils.get_scan_params(idx,self.allot_indices,R_i,self.dim)
    f = open(sfry_path+"/"+"submat"+str(R_i),'rb')
    f.seek(offset_in_bytes,0)
    rowbytes = f.read(readend - offset)
    return utils.query_exec(f.read(readend - offset), offset_correction, readend_correction, R_i, codebks, dim)
    

 
def query(word, word2idx, sfry_path): 
    
    dim = np.load(sfry_path+"/dim.npy")
    codebk = np.load(sfry_path+"/codebks.npy")
    allot_indices = np.load(sfry_path+"/metadata.npy")

    idx = word2idx[word]
    R_i = 0
    prev_index = 0
    while R_i < len(allot_indices):
        if idx >= allot_indices[R_i]:
            break
        else:
            R_i += 1

    if R_i == 0:
        return np.zeros(dim)
    
    f = open(sfry_path+"/"+"submat"+str(R_i),'rb')
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

    f.seek(offset_in_bytes,0)
    row_hex = f.read(readend_in_bytes - offset_in_bytes)
    row_bitstring = ""
    for i in range(0,len(row_hex)):
        bitstring = bin(row_hex[i])[2:]   
    #bitstring = bin(struct.unpack("B",row_hex[i])[0])[2:]
        if len(bitstring) < 8:
            bitstring = '0' * (8-len(bitstring)) + bitstring
        row_bitstring += bitstring 
    
    row_bitstring = row_bitstring[offset_correction:len(row_bitstring)-readend_correction]

    inflated_row = np.zeros(dim)
    for i in range(0,dim):
        code = int(row_bitstring[i*R_i:(i+1)*R_i],2)
        inflated_row[i] = codebk[R_i][code]
     
    return inflated_row
    

    
    
def compress(path, priorpath, mem_budget=None, R=1, write_inflated=False, word_rep="dict", write_word_rep=False):
    logging.basicConfig(filename=path+'.small-fry.log',level=logging.DEBUG)  
    logging.info("Initializing Small-Fry compression! Parameters: ")

   
    logging.info("Parsing embeddings txt and converting to npy...")
    emb_mat, p, words, word2idx, dim = utils.text2npy(path, priorpath, word_rep, write_word_rep)
    if mem_budget != None:
        mem_budget = float(mem_budget)
        R = 7.99*mem_budget/(len(p)*dim)
     
    logging.info("Computing optimal bit allocations...")
    bit_allocations = utils.allocation_round(utils.bit_allocator(p,R),sort=True)

    logging.info("Downsampling for dimension "+str(dim)+"...")
    bit_allocations = utils.downsample(bit_allocations, dim)

    logging.info("Computing submatrix partitions...") 
    submats,allot_indices = utils.mat_partition(emb_mat, bit_allocations)

    logging.info("Quantizing submatrices...")
    inflated_mat, quant_submats, codebks = utils.quantize(submats)
    infmat_path = path+".inflated.npy"
    np.save(infmat_path, inflated_mat)
    print("Saving Small-Fry representation to file...")
    sfry_path = utils.bitwrite_submats(quant_submats, codebks, path)
    np.save(sfry_path+"/codebks",codebks)
    np.save(sfry_path+"/metadata",allot_indices)
    np.save(sfry_path+"/dim",dim)
    print("Compression complete!!!")    

    return word2idx, sfry_path

parser = argh.ArghParser()
parser.add_commands([compress,query])

if __name__ == '__main__':
    parser.dispatch()


