from .import *
from .wrapper import *
import struct
import numpy as np
import argh
import logging
import re

def load(sfrypath, word2idx):
    return Smallfry(sfrypath, word2idx) 

def query(word, word2idx, sfrypath, usr_idx=False):
    '''
    Queries the Small-Fry representation for a word/index. 
    This can be imported as an API call or used as a command line utility.
        Params: 
        <word> should be the word string to query
        <word2idx> When used as API, should be a word representation that Small-Fry output, or
            alternatively the numeric index of the Small-Fry wordlist that is user managed. When used
            as command line utility, should be a sourcepath to the word representation or the numeric index
        <sfrypath> path to directory holding sfry 
        <usr_idx> Flag param, set True if word2idx is user-maintained word idx

    '''
    if usr_idx:
        word2idx = usr_idx_prep(word,int(word2idx))
    if type(word2idx) is str:
        word2idx = np.load(word2idx).item() 
    dim = np.load(sfrypath+"/metadata/dim.npy")
    allots = np.load(sfrypath+"/metadata/ballocs.npy")
    allot_indices = np.load("/metadata/ballocs_idx.npy")
    codebks = np.load(sfrypath+"/codebks.npy")
    idx, R_i, OofV = query_prep(word, word2idx, dim, codebks, allot_indices)
    if R_i == 0:    
        return OofV
    offset, readend, offset_correction, readend_correction = get_scan_params(idx,allot_indices,R_i,dim)
    f = open(sfrypath+"/"+"submat"+str(R_i),'rb')
    f.seek(offset,0)
    return query_exec(f.read(readend - offset), offset_correction, readend_correction, R_i, codebks, dim)
    
    
def compress(sourcepath, priorpath, outdir=None, mem_budget=None, R=1, write_inflated=False, word_rep="dict", write_word_rep=False):
    '''
    Compresses the source embeddings. 
    This can be imported as an API call or used as a command line utility.
        Params:
            <sourcepath> path to source embeddings
            <priorsourcepath> path to prior in for as npy dict
            <mem_budget> provides an approximate memory budget
            <R> provide a bitrate constraint 
            <write_inflated> Flag for writing inflated embeddings matrix in npy format  
            <word_rep> Should be either "dict" or "trie" 
            <write_word_rep> Flag -- if set low will output a text word list in index order. 
                User is responsible for managing the word representation

    '''
    logging.basicConfig(filename=sourcepath+'.small-fry.log',level=logging.DEBUG)  
    logging.info("Initializing Small-Fry compression! Parameters: ")
 
    logging.info("Parsing embeddings txt and converting to npy...")
    outpath = ""
    if outdir == None:
        outpath= sourcepath
    else:
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        outpath = outdir+"/"+(re.split("/",sourcepath)[-1])

    emb_mat, p, words, word2idx, dim = text2npy(sourcepath, outpath, priorpath, word_rep, write_word_rep)
    if mem_budget != None:
        mem_budget = float(mem_budget)
        R = 7.99*mem_budget/(len(p)*dim)
     
    logging.info("Computing optimal bit allocations...")
    bit_allocations = allocation_round(bit_allocator(p,R),sort=True)

    logging.info("Downsampling for dimension "+str(dim)+"...")
    bit_allocations = downsample(bit_allocations, dim)

    logging.info("Computing submatrix partitions...") 
    submats, allots, allot_indices = mat_partition(emb_mat, bit_allocations)
    
    submats, allots, allot_indices = matpart_adjuster(submats, allots, allot_indices, len(words))

    logging.info("Quantizing submatrices...")
    inflated_mat, quant_submats, codebks = quantize(submats)
    if write_inflated:
        logging.info("Writing inflated embeddings as npy...")
        npy2text(inflated_mat, words, outpath+".inflated")
    print("Saving Small-Fry representation to file...")
    sfry_path = bitwrite_submats(quant_submats, codebks, outpath)
    np.save(sfry_path+"/codebks",codebks)
    meta_path = sfry_path+"/metadata"
    os.mkdir(meta_path)
    np.save(meta_path+"/ballocs",allots)
    np.save(meta_path+"/dim",dim)
    np.save(meta_path+"/ballocs_idx",allot_indices)
    print("Compression complete!!!")    

    return word2idx, sfry_path

#parser = argh.ArghParser()
#parser.add_commands([compress,query])

#if __name__ == '__main__':
#    parser.dispatch()


