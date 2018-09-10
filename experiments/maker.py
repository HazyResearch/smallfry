import json
import pathlib
import numpy as np
import os
import argparse
import datetime
import logging
import time
from subprocess import check_output
from neuralcompressor.nncompress import EmbeddingCompressor
from smallfry.smallfry import Smallfry
from smallfry import utils as sfry_utils

def main():
    config = vars(init_parser().parse_args())
    assert int(np.log2(config['k'])) == np.log2(config['k']),\
        'k must be a power of two.'

    # load base embeddings
    base_embeds, wordlist = sfry_utils.load_embeddings(config['basepath'])
    (v,d) = base_embeds.shape
    assert len(wordlist) == v, 'Embedding dim must match wordlist length.'

    # update config
    config['vocab'] = v
    config['dim'] = d
    config['githash'] = get_git_hash()
    config['date'] = get_date_str()
    config['date-rungroup'] = '{}-{}'.format(config['date'],config['rungroup'])
    config['memory'] = get_memory(config)
    config['bitrate'] = config['memory'] / v * d
    config['compression-ratio'] = 32 * v * d / config['memory']

    # Make embeddings
    embed_dir, embed_name = get_embeddings_dir_and_name(config)
    core_filename = str(pathlib.PurePath(embed_dir, embed_name))
    os.makedirs(embed_dir)
    init_logging(core_filename + '_maker.log')
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
    return parser

def init_logging(log_filename):
    """Initialize logfile to be used for experiment."""
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(message)s',
                        datefmt='[%m/%d/%Y %H:%M:%S]: ',
                        filemode='w', # this will overwrite existing log file.
                        level=logging.DEBUG)
    logging.info('Begin logging.')

def make_embeddings(base_embeds, embed_dir, config):
    if config['method'] == 'kmeans':
        sfry = Smallfry.quantize(base_embeds, b=config['bitsperblock'],
            block_len=config['blocklen'], r_seed=config['seed'])
        embeds = sfry.decode(np.array(list(range(config['vocab']))))
    elif config['method'] == 'dca':
        m,k,v,d = config['m'], config['k'], config['vocab'], config['dim']
        work_dir = str(pathlib.PurePath(embed_dir,'dca_tmp'))
        compressor = EmbeddingCompressor(m, k, work_dir)
        base_embeds = base_embeds.astype(np.float32)
        compressor.train(base_embeds)
        codes, codebook = compressor.export(base_embeds, work_dir)
        codes = np.array(codes).flatten()
        codebook = np.array(codebook)
        embeds = codes_2_vec(codes, codebook, m, k, v, d)
    else:
        raise ValueError('Method name invalid')
    return embeds

def get_embeddings_dir_and_name(config):
    if config['method'] == 'kmeans':
        params = ['base','vocab','dim','bitsperblock','blocklen','seed','date', 'rungroup']
    elif config['method'] == 'dca':
        params = ['base','vocab','dim','m','k','seed','date', 'rungroup']
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

def codes_2_vec(codes, codebook, m, k, v, d):
    codes = codes.reshape(int(len(codes)/m),m)
    dcc_mat = np.zeros([v,d])
    for i in range(0,v):
        for m in range(0,m):
            dcc_mat[i,:] += codebook[m*k+codes[i,m],:]
    return dcc_mat

def get_date_str():
	return '{:%Y-%m-%d}'.format(datetime.date.today())

def to_file_np(path, embeds):
    np.save(path, embeds)

def to_file_txt(path, wordlist, embeds):
    with open(path, "w+") as file:
        for i in range(len(wordlist)):
            file.write(wordlist[i] + " ")
            row = embeds[i, :]
            strrow = [str(r) for r in row]
            file.write(" ".join(strrow))
            file.write("\n")

def get_git_hash():
   git_hash = None
   try:
       git_hash = check_output(['git','rev-parse','--short','HEAD']).strip()
       logging.info('Git hash {}'.format(git_hash))
   except FileNotFoundError:
       logging.info('Unable to get git hash.')
   return str(git_hash)


def save_dict_as_json(dict_to_write, path):
    with open(path, 'w') as f: json.dump(dict_to_write, f, indent=2)

if __name__ == '__main__':
    main()
