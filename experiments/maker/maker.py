import json
import pathlib
import os
import argparse
import logging
import sys
import numpy as np
from subprocess import check_output
from smallfry.smallfry import Smallfry
from smallfry.utils import load_embeddings
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
    logging.info('Begining to make embeddings')
    start = time.time()
    embeds = make_embeddings(base_embeds, embed_dir, config)
    end = time.time()
    maketime = end - start
    logging.info('Finished making embeddings.'
                 'It took {} minutes'.format(maketime/60))
    config['maketime-secs'] = maketime


    # Save embeddings (text and numpy) and config
    to_file_txt(core_filename + '.txt', wordlist, embeds)
    to_file_np(core_filename + '.npy', embeds)
    save_dict_as_json(config, core_filename + '_config.json')
    logging.info('maker.py finished!')

def init_parser():
    """Initialize Cmd-line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
        choices=['kmeans','dca'],
        help='Name of compression method to use (kmeans or dca).')
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
    parser.add_argument('--bitsperblock', type=int,
        help='Bits per block')
    parser.add_argument('--blocklen', type=int, default=1,
        help='Block length for quantization/k-means')
    parser.add_argument('--m', type=int, default=1,
        help='Number of codebooks for DCA.')
    parser.add_argument('--k', type=int, default=1,
        help='Codebook size for DCA.')
    parser.add_argument('--ibr', type=float, required=True,
        help='Developer intended bitrate.')
    return parser

def init_logging(log_filename):
    """Initialize logfile to be used for experiment."""
    print('logging')
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(message)s',
                        datefmt='[%m/%d/%Y %H:%M:%S]: ',
                        filemode='w', # this will overwrite existing log file.
                        level=logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)
    logging.info('Begin logging.')

def make_embeddings(base_embeds, embed_dir, config):
    if config['method'] == 'kmeans':
        assert config['bitsperblock']/config['blocklen'] == config['ibr'], "intended bitrate for kmeans not met!"
        sfry = Smallfry.quantize(base_embeds, b=config['bitsperblock'],
            block_len=config['blocklen'], r_seed=config['seed'])
        embeds = sfry.decode(np.array(list(range(config['vocab']))))
    elif config['method'] == 'dca':
        m,k,v,d,ibr = config['m'], config['k'], config['vocab'], config['dim'], config['ibr']
        #does m make sense given ibr and k?
        assert m == compute_m_dca(k,v,d,ibr), "m and k does not match intended bit rate"
        work_dir = str(pathlib.PurePath(embed_dir,'dca_tmp'))
        compressor = EmbeddingCompressor(m, k, work_dir)
        base_embeds = base_embeds.astype(np.float32)
        dca_train_log = compressor.train(base_embeds)
        me_distance,frob_error = compressor.evaluate(base_embeds)
        config['mean-euclidean-dist'] = me_distance
        config['embed-frob-err'] = frob_error
        with open(work_dir+'.dca-log-json','w+') as log_f:
            log_f.write(json.dumps(dca_train_log))
        codes, codebook = compressor.export(base_embeds, work_dir)
        codes = np.array(codes).flatten()
        codebook = np.array(codebook)
        embeds = codes_2_vec(codes, codebook, m, k, v, d)
    else:
        raise ValueError('Method name invalid')
    return embeds

def get_embeddings_dir_and_name(config):
    if config['method'] == 'kmeans':
        params = ['base','method','vocab','dim','ibr','bitsperblock','blocklen','seed','date','rungroup']
    elif config['method'] == 'dca':
        params = ['base','method','vocab','dim','ibr','m','k','seed','date','rungroup']
    else:
        raise ValueError('Method name invalid')

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
    else:
        raise ValueError('Method name invalid (must be dca or kmeans)')
    return mem

if __name__ == '__main__':
    main()
