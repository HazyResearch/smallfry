import json
import pathlib
import os
import argparse
import logging
import sys
import datetime
import numpy as np
from subprocess import check_output
from smallfry.smallfry import Smallfry
from smallfry.utils import load_embeddings
from uniform_quant import *
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) #FOR LOCAL IMPORTS
from experimental_utils import * 
from neuralcompressor.nncompress import EmbeddingCompressor


def main():
    config = vars(init_parser().parse_args())
    assert int(np.log2(config['k'])) == np.log2(config['k']),\
        'k must be a power of two.'

    # in order to keep things clean, rungroups should not have underscores:
    assert '_' not in config['rungroup'], 'rungroup names should not have underscores'

    # load base embeddings
    base_embeds, wordlist = load_embeddings(config['basepath'])
    (v,d) = base_embeds.shape
    assert len(wordlist) == v, 'Embedding dim must match wordlist length.'

    # set random seed
    set_seeds(config['seed'])

    # update config
    config['vocab'] = v
    config['dim'] = d
    config['date'] = get_date_str()
    config['date-rungroup'] = '{}-{}'.format(config['date'],config['rungroup'])
    config['memory'] = get_memory(config)
    config['bitrate'] = config['memory'] / (v * d)
    config['compression-ratio'] = 32 * v * d / config['memory']

    # Make embeddings
    embed_dir, embed_name = get_embeddings_dir_and_name(config)
    core_filename = str(pathlib.PurePath(embed_dir, embed_name))
    os.makedirs(embed_dir)
    init_logging(core_filename + '_maker.log')
    config['githash-maker'] = get_git_hash()
    logging.info(f"Begining to make embeddings: {config}")
    logging.info('Datetime is %s' % datetime.datetime.now())
    start = time.time()
    embeds = make_embeddings(base_embeds, embed_dir, config)
    end = time.time()
    maketime = end - start
    logging.info('Datetime is %s' % datetime.datetime.now())
    logging.info('Finished making embeddings.'
                 'It took {} minutes'.format(maketime/60))
    config['maketime-secs'] = maketime
    config['range'] = str(np.max(np.abs(embeds))) #add range 

    # Save embeddings (text and numpy) and config
    to_file_txt(core_filename + '.txt', wordlist, embeds)
    if config['writenpy']:
        to_file_np(core_filename + '.npy', embeds)
    save_dict_as_json(config, core_filename + '_config.json')
    logging.info('maker.py finished!')

def init_parser():
    """Initialize Cmd-line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
        choices=['kmeans','dca','baseline','stochround','midriser','optranuni','clipnoquant','stochoptranuni','naiveuni','stoch_range_2'],
        help='Name of compression method to use (kmeans or dca or stochastic rounding).')
    parser.add_argument('--base', type=str, required=True,
        help='Name of base embeddings')
    parser.add_argument('--basepath', type=str, required=True,
        help='Path to base embeddings')
    parser.add_argument('--seed', type=int, required=True,
        help='Random seed to use for experiment.')
    parser.add_argument('--outputdir', type=str, required=True,
        help='Head output directory')
    parser.add_argument('--rungroup', type=str, required=True,
        help='Rungroup for organization')
    parser.add_argument('--ibr', type=float, required=True,
        help='Developer intended bitrate.')
    parser.add_argument('--bitsperblock', type=int,
        help='Bits per block')
    parser.add_argument('--blocklen', type=int, default=1,
        help='Block length for quantization/k-means')
    parser.add_argument('--m', type=int, default=1,
        help='Number of codebooks for DCA.')
    parser.add_argument('--k', type=int, default=1,
        help='Codebook size for DCA.')
    parser.add_argument('--tau', type=float, default=1.0,
        help='Temperature parameter for deep net training.')
    parser.add_argument('--batchsize', type=int, default=64,
        help='Batch size for deep net training.')
    parser.add_argument('--gradclip', type=float, default=0.001,
        help='Clipping of gradient norm for deep net training.')
    parser.add_argument('--lr', type=float, default=0.0001,
        help='Learning rate for deep net training.')
    parser.add_argument('--solver', type=str, default='iterative',
        choices=['iterative','dynprog'],
        help='Solver used to solve k-means.')
    parser.add_argument('--writenpy', type=bool, default=False,
        help='Write embeddings matrix in npy format in addition to text')
    return parser

def make_embeddings(base_embeds, embed_dir, config):
    if config['method'] == 'kmeans':
        assert config['bitsperblock']/config['blocklen'] == config['ibr'], "intended bitrate for kmeans not met!"
        start = time.time()
        sfry = Smallfry.quantize(base_embeds, b=config['bitsperblock'], solver=config['solver'],
            block_len=config['blocklen'], r_seed=config['seed'])
        config['sfry-maketime-quantize-secs'] = time.time()-start
        config['embed-maketime-secs'] = config['sfry-maketime-quantize-secs']
        start = time.time()
        embeds = sfry.decode(np.array(list(range(config['vocab']))))
        config['sfry-maketime-decode-secs'] = time.time()-start
    elif config['method'] == 'dca':
        m,k,v,d,ibr = config['m'], config['k'], config['vocab'], config['dim'], config['ibr']
        #does m make sense given ibr and k?
        assert m == compute_m_dca(k,v,d,ibr), "m and k does not match intended bit rate"
        work_dir = str(pathlib.PurePath(embed_dir,'dca_tmp'))
        start = time.time()
        compressor = EmbeddingCompressor(m, k, work_dir, 
                                            config['tau'],
                                            config['batchsize'],
                                            config['lr'],
                                            config['gradclip'])
        base_embeds = base_embeds.astype(np.float32)
        dca_train_log = compressor.train(base_embeds)
        config['dca-maketime-train-secs'] = time.time()-start
        config['embed-maketime-secs'] = config['dca-maketime-train-secs']
        me_distance,frob_error = compressor.evaluate(base_embeds)
        config['mean-euclidean-dist'] = me_distance
        config['embed-frob-err'] = frob_error
        with open(work_dir+'.dca-log-json','w+') as log_f:
            log_f.write(json.dumps(dca_train_log))
        start = time.time()
        codes, codebook = compressor.export(base_embeds, work_dir)
        config['dca-maketime-export-secs'] = time.time() - start
        start = time.time()
        codes = np.array(codes).flatten()
        codebook = np.array(codebook)
        embeds = codes_2_vec(codes, codebook, m, k, v, d)
        config['dca-maketime-codes2vec-secs'] = time.time() - start
    elif config['method'] == 'baseline':
        assert config['ibr'] == 32.0, "Baselines use floating point precision"
        embeds = load_embeddings(config['basepath'])[0]
    elif config['method'] == 'clipnoquant':
        embeds = load_embeddings(config['basepath'])[0]
        start = time.time()
        embeds = clip_no_quant(embeds,config['ibr'])
        config['embed-maketime-secs'] = time.time()-start
        config['embed-fro-dist'] = np.linalg.norm(base_embeds - embeds)
    elif config['method'] == 'midriser':
        embeds = load_embeddings(config['basepath'])[0]
        start = time.time()
        embeds = midriser(base_embeds,config['ibr'])
        config['embed-maketime-secs'] = time.time()-start
    elif config['method'] == 'naiveuni':
        embeds = load_embeddings(config['basepath'])[0]
        start = time.time()
        embeds = naiveuni(base_embeds,config['ibr'])
        config['embed-maketime-secs'] = time.time()-start
    elif config['method'] == 'stochround':
        embeds = load_embeddings(config['basepath'])[0]
        start = time.time()
        embeds = stochround(base_embeds,config['ibr'])
        config['embed-maketime-secs'] = time.time()-start
    elif config['method'] == 'optranuni':
        embeds = load_embeddings(config['basepath'])[0]
        start = time.time()
        embeds = adarange(base_embeds,config['ibr'])
        config['embed-maketime-secs'] = time.time()-start
        config['embed-fro-dist'] = np.linalg.norm(base_embeds - embeds)
    elif config['method'] == 'stochoptranuni':
        embeds = load_embeddings(config['basepath'])[0]
        start = time.time()
        embeds = stoch_adarange(base_embeds,config['ibr'])
        config['embed-maketime-secs'] = time.time()-start
        config['embed-fro-dist'] = np.linalg.norm(base_embeds - embeds)
        #TODO remove this from here
    elif config['method'] == 'stoch_range_2':
        embeds = load_embeddings(config['basepath'])[0]
        start = time.time()
        embeds = stoch_adarange_2(base_embeds,config['ibr'])
        config['embed-maketime-secs'] = time.time()-start
        config['embed-fro-dist'] = np.linalg.norm(base_embeds - embeds)
        #TODO remove this from here
    else:
        raise ValueError(f"Method name invalid {config['method']}")
    return embeds

def get_embeddings_dir_and_name(config):
    if config['method'] == 'kmeans':
        params = ['base','method','vocab','dim','ibr','bitsperblock','blocklen','seed','date','rungroup','solver']
    elif config['method'] == 'dca':
        params = ['base','method','vocab','dim','ibr','m','k','seed','date','rungroup','lr','gradclip','batchsize','tau']
    elif config['method'] in ['baseline', 'stochround', 'naiveuni', 'midriser','optranuni','clipnoquant','stochoptranuni','stoch_range_2']:
        params = ['base','method','vocab','dim','ibr','seed','date','rungroup']
    else:
        raise ValueError(f"Method name invalid {config['method']}")

    embed_name_parts = ['{}={}'.format(param, config[param]) for param in params]
    embed_name = ','.join(embed_name_parts)
    embed_dir = str(pathlib.PurePath(
            config['outputdir'],
            config['date-rungroup'],
            embed_name
        ))
    return embed_dir, embed_name

def get_memory(config):
    v = config['vocab']
    d = config['dim']
    if config['method'] == 'kmeans':
        bitsperblock = config['bitsperblock']
        blocklen = config['blocklen']
        num_blocks = d / blocklen
        num_centroids = 2**bitsperblock
        mem = v * num_blocks * bitsperblock + num_centroids * blocklen * 32
    elif config['method'] == 'dca':
        m = config['m']
        k = config['k']
        mem = v * m * np.log2(k) + 32 * d * m * k
    elif config['method'] in ['baseline','clipnoquant']:
        return v*d*32
    elif config['method'] in ['stochround', 'naiveuni', 'midriser','optranuni','stochoptranuni','stoch_range_2'] :
        return config['ibr']*d*v + 32
    else:
        raise ValueError('Method name invalid (must be dca or kmeans)')
    return mem


def golden_section_search(f, x_min=1e-10, x_max=10, tol=1e-2):
    '''
    Find argmin of f between x_min and x_max (for f uni-modal).
    
    This function uses the golden-section search algorithm.
    It always maintains a list of four points [x1,x2,x3,x4],
    which are always spaced as: [a,a+(c^2)h,a+ch,a+h].
    for c = (math.sqrt(5) - 1) / 2 = 0.618...
    The algorithm progressively reduces the size of the interval being
    considered by checking whether f(x2) < f(x3), and eliminating one of the
    endpoints accordingly; x4 is eliminated if f(x2) < f(x3), and x1 
    is eliminated otherwise.
    
    If f(a+(c^2)h) < f(a+ch), the new interval becomes
    >>> [a,a+(c^3)h,a+(c^2)h,a+ch] = [a,a+(c^2)(ch),a+c(ch),a+ch]
    (So h' = ch, a' = a)
    Otherwise, the interval becomes
    >>> [a',a'+(c^2)h',a'+ch', a'+h'], for a' = a+(c^2)h and h'=(h-(c^2)h)
    It is easy to check that a'+(c^2)h' = a + ch, and that a'+h' = a+h,
    So this interval is equal to [a+(c^2)h, a+ch, X, a+h], for X=a'+ch'

    The algorithm terminates when it has been narrowed
    down that the argmin must be in an interval of size < tol.
    '''
    #initialize points
    phi = (math.sqrt(5) - 1) / 2
    x1 = x_min
    x4 = x_max
    f_x1 = f(x1)
    f_x4 = f(x4)
    x2 = x1 + (x4-x1) * phi**2
    x3 = x1 + (x4-x1) * phi
    f_x2 = f(x2)
    f_x3 = f(x3)
    while (x4-x1 > tol):
        assert (math.isclose(x2, x1 + (x4 - x1) * phi**2) and 
                math.isclose(x3, x1 + (x4 - x1) * phi))
        if f_x2 < f_x3:
            # The new points become [x1, NEW, x2, x3]
            x4,f_x4 = x3,f_x3
            x3,f_x3 = x2,f_x2
            x2 = x1 + (x4-x1) * phi**2
            f_x2 = f(x2)
        else:
            # The new points become [x2, x3, NEW, x4]
            x1,f_x1 = x2,f_x2
            x2,f_x2 = x3,f_x3
            x3 = x1 + (x4-x1) * phi
            f_x3 = f(x3)
        
    # Return x-value with minimum f(x) which was found.
    i = np.argmin([f_x1,f_x2,f_x3,f_x4])
    x = [x1,x2,x3,x4]
    return x[i]



if __name__ == '__main__':
    main()
