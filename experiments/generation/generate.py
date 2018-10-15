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
        if config['writenpy']:
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
        choices=['text8','wiki.en.txt'],
        help='Natural language dataset used to train embeddings')
    parser.add_argument('--seed', type=int, required=True,
        help='Random seed to use for experiment.')
    parser.add_argument('--outputdir', type=str, required=True,
        help='Head output directory')
    parser.add_argument('--rungroup', type=str, required=True,
        help='Rungroup for organization')
    parser.add_argument('--dim', type=int, default=300,
        help='Dimension for generated embeddings')
    parser.add_argument('--maxvocab', type=int, default=400000,
        help='Maximum vocabulary size')
    parser.add_argument('--memusage', type=int, default=256,
        help='Memory usage in GB')
    parser.add_argument('--numthreads', type=int, default=52,
        help='Number of threads to spin up')
    parser.add_argument('--numiters', type=int, default=50,
        help='Number of iterations to optimize over')
    parser.add_argument('--writenpy', type=bool, default=False,
        help='Write embeddings matrix in npy format in addition to text')
    parser.add_argument('--windowsize', type=int, default=15,
        help='Window size for use in co-oc calculations')
    parser.add_argument('--vocabmincount', type=int, default=5,
        help='Minimum oc count for a vocab')
    return parser

def generate_embeddings(config, embed_dir, embed_name):
    embeds = None #optional populate
    wordlist = None #optional populate (but required if embeds is populated)
    v = None #this value must be populated by all method types
    if config['method'] == 'glove':
        gen_glove_qry = str(pathlib.PurePath(get_glove_generator_path(), '*' ))
        cp_stuff_2_dir = f"cp -r {gen_glove_qry} {embed_dir}"
        logging.info(cp_stuff_2_dir)
        os.system(cp_stuff_2_dir)
        os.chdir(embed_dir)
        #check to see if co-oc-shuf already exists:
        corpuspath = str(pathlib.PurePath( get_corpus_path(), config['corpus']))
        coocshufpath = f"{corpuspath}.maxvocab_{config['maxvocab']}.windowsize_{config['windowsize']}.seed_{config['seed']}.vocabmincount_{config['vocabmincount']}.memory_{config['memusage']}.cooccurrence.shuf.bin"
        logging.info(f"Co-oc shuf patg: {coocshufpath}")
        cooc_exists_bool = os.path.isfile(coocshufpath)
        if cooc_exists_bool:
            cooc_exists = 1 
            coocpath = None #command will directly use cooc-shuf
            vocabpath = f"{corpuspath}.vocab.txt"
        else:
            cooc_exists = 0
            coocshufpath = "cooccurrence.shuf.bin"
            coocpath = "cooccurrence.bin"
            vocabpath = "vocab.txt"

        logging.info(f"Does co-oc exist? {cooc_exists_bool}")
        #check gen_glove.sh to get correct ORDER for these arguments
        output = perform_command_local(f"bash gen_glove.sh {cooc_exists} {corpuspath} {vocabpath} {coocpath} {coocshufpath} {config['memusage']} {config['dim']} {config['maxvocab']} {config['numiters']} {config['windowsize']} {config['numthreads']} {config['seed']} {config['vocabmincount']} {embed_name}")
        logging.info(output)
        wc = perform_command_local(f"wc -l {embed_name}.txt")
        logging.info(wc)
        v = int(wc.split(' ')[0])
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
