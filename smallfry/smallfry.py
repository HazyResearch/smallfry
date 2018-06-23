from .core import text2npy, npy2text, allocation_round, downsample, quantize
from .core import mat_partition, matpart_adjuster, bitwrite_submats, bit_allocator

import os, logging, argh
import json
import numpy as np

def load(sfrypath, word2idx):
    return Smallfry(sfrypath, word2idx) 


# The use of membudget and r is confusing. Maybe split into two interfacces?
@argh.arg("source_file", help="The source embedding file")
@argh.arg("prior_file", help="The priors in Numpy Format")
@argh.arg("outdir", help="Output directory")
@argh.arg("-R", help="compression rate")
@argh.arg("--write-inflated", help="Write a numpy version")
@argh.arg("--word-rep", help="Valid values are {trie, dict}")
def compress(source_file, 
             prior_file, 
             outdir, 
             mem_budget=None, 
             R=1.0, 
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
            <write_inflated> Output an inflated embeddings matrix in format  
            <word_rep> Should be either "dict" or "trie" 
            <write_word_rep> Flag -- if set low will output a text word list in index order. 
                User is responsible for managing the word representation

    '''
    assert not os.path.isdir(outdir), f"User specified output directory already exists ({outdir})"
    os.mkdir(outdir)

    logging.basicConfig(format="%(asctime)s %(message)s",
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filename=os.path.join(outdir, 'small-fry.log'),level=logging.DEBUG)  
    logging.info("Initializing Small-Fry compression! Parameters: ")    
    
    emb_mat, p, words, word2idx, dim = text2npy(source_file, outdir, prior_file, word_rep)
    if mem_budget != None:
        mem_budget = float(mem_budget)
        R = 7.99*mem_budget/(len(p)*dim)
     
    logging.info("Computing optimal bit allocations...")
    bit_allocations = allocation_round(bit_allocator(p,R))

    logging.info(f"Downsampling for dimension {dim}...")
    bit_allocations = downsample(bit_allocations, dim, max_bitrate, sampling_topdwn)

    logging.info("Computing submatrix partitions...")
    # This is very strange code
    # Maybe it should be wrapped in a single method?
    submats, allots, allot_indices = mat_partition(emb_mat, bit_allocations)    
    submats, allots, allot_indices = matpart_adjuster(submats, allots, allot_indices, len(p))

    logging.info("Quantizing submatrices...")
    inflated_mat, quant_submats, codebks = quantize(submats, allots, minibatch)

    if write_inflated:
        path_out = os.path.join(outdir, "sfry.inflated.glove")
        logging.info(f"Writing inflated embeddings into {path_out} as npy...")
        npy2text(inflated_mat, words, path_out)
  
    bitwrite_submats(quant_submats, codebks, allots, outdir)
    print("Saving Small-Fry representation to file: " + str(outdir))
    np.save(os.path.join(outdir, "codebks"),codebks)
    metadata = dict()
    metadata['allots'] = [int(a) for a in allots]
    metadata['dim'] = dim
    metadata['allot_indices'] = [int(a_i) for a_i in allot_indices]
    json.dump(metadata, open(os.path.join(outdir, "metadata"),'w'))

    print("Compression complete!!!")   
    #return word2idx, outdir

parser = argh.ArghParser()
parser.add_commands([compress])
if __name__ == '__main__':
    parser.dispatch()
