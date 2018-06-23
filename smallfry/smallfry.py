from .import *
from .wrapper import *
import struct
import numpy as np
import argh
import logging
import re
import json

def load(sfrypath, word2idx):
    return Smallfry(sfrypath, word2idx) 
    
def compress(sourcepath, 
             priorpath, 
             outdir, 
             mem_budget=None, 
             R=1, 
             write_inflated=False, 
             word_rep="trie", 
             minibatch=False,
             max_bitrate=None, 
             sampling_topdwn=True,
             ):
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
    
    assert not os.path.isdir(outdir), "User specified output directory already exists"
    os.mkdir(outdir)

    emb_mat, p, words, word2idx, dim = text2npy(sourcepath, outdir, priorpath, word_rep)
    if mem_budget != None:
        mem_budget = float(mem_budget)
        R = 7.99*mem_budget/(len(p)*dim)
     
    logging.info("Computing optimal bit allocations...")
    bit_allocations = allocation_round(bit_allocator(p,R),sort=True)

    logging.info("Downsampling for dimension "+str(dim)+"...")
    bit_allocations = downsample(bit_allocations, dim, max_bitrate, sampling_topdwn)

    logging.info("Computing submatrix partitions...") 
    submats, allots, allot_indices = mat_partition(emb_mat, bit_allocations)
    
    submats, allots, allot_indices = matpart_adjuster(submats, allots, allot_indices, len(p))

    logging.info("Quantizing submatrices...")
    inflated_mat, quant_submats, codebks = quantize(submats, allots, minibatch)

    if write_inflated:
        logging.info("Writing inflated embeddings as npy...")
        npy2text(inflated_mat, words, outdir+"/sfry.inflated.txt")
  
    bitwrite_submats(quant_submats, codebks, allots, outdir)
    print("Saving Small-Fry representation to file: " + str(outdir))
    np.save(outdir+"/codebks",codebks)
    metadata = dict()
    metadata['allots'] = [int(a) for a in allots]
    metadata['dim'] = dim
    metadata['allot_indices'] = [int(a_i) for a_i in allot_indices]
    json.dump(metadata, open(outdir+"/metadata",'w'))

    print("Compression complete!!!")    

    return word2idx, outdir

