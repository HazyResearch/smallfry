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

def query(word, word2idx, sfry_path):
    if type(word2idx) is str:
        word2idx = np.load(word2idx).item() 
    dim = np.load(sfry_path+"/dim.npy")
    codebks = np.load(sfry_path+"/codebks.npy")
    allot_indices = np.load(sfry_path+"/metadata.npy")
    idx, R_i, OofV = utils.query_prep(word, word2idx, dim, codebks, allot_indices)
    if R_i == 0:    
        return OofV
    offset, readend, offset_correction, readend_correction = utils.get_scan_params(idx,allot_indices,R_i,dim)
    f = open(sfry_path+"/"+"submat"+str(R_i),'rb')
    f.seek(offset,0)
    return utils.query_exec(f.read(readend - offset), offset_correction, readend_correction, R_i, codebks, dim)
    
    
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
    if write_inflated:
        logging.info("Writing inflated embeddings as npy...")
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


