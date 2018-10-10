import json
import pathlib
import os
import argparse
import logging
import sys
import numpy as np
from subprocess import check_output
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) #FOR LOCAL IMPORTS
from experimental_utils import * 

def main():
    config = vars(init_parser().parse_args())

    # in order to keep things clean, rungroups should not have underscores:
    assert '_' not in config['rungroup'], 'rungroup names should not have underscores'

    # update config
    config['date'] = get_date_str()
    config['date-rungroup'] = '{}-{}'.format(config['date'],config['rungroup'])
    print(config)

    # Gen embeddings
    embed_dir, embed_name = get_embeddings_dir_and_name(config)
    core_filename = str(pathlib.PurePath(embed_dir, embed_name)) #full path to embeddings txt
    os.makedirs(embed_dir)
    init_logging(core_filename + '_maker.log')
    #TODO ADD BACK IN THIS LATER config['githash-maker'] = get_git_hash()
    logging.info('Begining to gen embeddings')
    start = time.time()
    embeds, wordlist, v = generate_embeddings(config, embed_dir, embed_name) #this routine must return v to set config
    end = time.time()
    maketime = end - start
    logging.info('Finished generating embeddings.'
                 'It took {} minutes'.format(maketime/60))

    #update config post embeddings generation
    config['gentime-secs'] = maketime
    config['vocab'] = v
    config['memory'] = get_memory(config)
    config['bitrate'] = config['memory'] / (config['vocab'] * config['dim'])
    config['compression-ratio'] = 32 * config['vocab'] * config['dim'] / config['memory']

    # Save embeddings (text and numpy) and config
    if not embeds == None:
        assert not wordlist == None, f"Generate.py requires both embeds and wordlists to write to file! Offending method: {config['method']}"
        logging.info(f"For method {config['method']}, generate.py is responsible for writing embeddings to file...")
        to_file_txt(core_filename + '.txt', wordlist, embeds)
        to_file_np(core_filename + '.npy', embeds)
        save_dict_as_json(config, core_filename + '_config.json')
        logging.info(f"Write complete")
    logging.info('maker.py finished!')

def init_parser():
    """Initialize Cmd-line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
        choices=['glove'],
        help='Name of embeddings training algorithm.')
    parser.add_argument('--corpus', type=str, required=True,
        choices=['text8'],
        help='Natural language dataset used to train embeddings')
    parser.add_argument('--seed', type=int, required=True,
        help='Random seed to use for experiment.')
    parser.add_argument('--outputdir', type=str, required=True,
        help='Head output directory')
    parser.add_argument('--rungroup', type=str, required=True,
        help='Rungroup for organization')
    parser.add_argument('--dim', type=int, default=300,
        help='Dimension for generated embeddings')
    parser.add_argument('--maxvocab', type=int, default=1e5,
        help='Maximum vocabulary size')
    parser.add_argument('--memusage', type=int, default=128,
        help='Memory usage in GB')
    parser.add_argument('--numthreads', type=int, default=24,
        help='Number of threads to spin up')
    return parser

def generate_embeddings(config, embed_dir, embed_name):
    embeds = None #optional populate
    wordlist = None #optional populate (but required if embeds is populated)
    v = None #this value must be populated by all method types
    if config['method'] == 'glove':
        gen_glove_qry = str(pathlib.PurePath(get_glove_generator_path(), '/*' ))
        os.system(f"cp {gen_glove_qry} {embed_dir}")
        os.chdir(embed_dir)
        corpuspath = str(pathlib.PurePath( get_corpus_path(), config['corpus']))
        output = os.system(f"bash gen_glove.sh {corpuspath} \
                                    {config['dim']} \
                                    {config['maxvocab']} \
                                    {config['numthreads']} \
                                    {config['memusage']} \
                                    {embed_name}")
        logging.info(output)
        wc = perform_command_local(f"wc {embed_name}.txt")
        v = wc.split(' ')[0]
    else:
        raise ValueError(f"Method name invalid: {config['method']}")
    assert not v == None, f"Method {config['method']} does must return vocab size"
    return embeds, wordlist, v

def get_embeddings_dir_and_name(config):
    if config['method'] == 'glove':
        params = ['corpus','method','maxvocab','dim','memusage','seed','date','rungroup']
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
    if config['method'] == 'glove':
       mem = v*d*32
    else:
        raise ValueError('Method name invalid (must be dca or kmeans)')
    return mem

if __name__ == '__main__':
    main()
