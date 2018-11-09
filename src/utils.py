import os
import sys
import socket
import json
import datetime
import logging
import pathlib
import time
import random
from subprocess import check_output
import argparse
import numpy as np
import torch
import tensorflow as tf

config = {}

def init(runtype):
    if runtype == 'train':
        parser = init_train_parser()
    elif runtype == 'compress':
        parser = init_compress_parser()
    elif runtype == 'evaluate':
        parser = init_evaluate_parser()
    init_config(parser, runtype)
    init_random_seeds()

def init_train_parser():
    """Initialize cmd-line parser for embedding training."""
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
    parser.add_argument('--lr', type=float, default=0.05,
        help='Learning rate for embedding training.')
    return parser

def init_compress_parser():
    """Initialize cmd-line parser for embedding compression."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedtype', type=str, required=True,
        choices=['glove400k'], # TODO: Add more options
        help='Name of embedding to compress')
    parser.add_argument('--embeddim', type=int, default=300,
        help='Dimension of embeddings to use.')
    parser.add_argument('--compresstype', type=str, required=True,
        choices=['uniform','kmeans','dca','nocompress'],
        help='Name of compression method to use (uniform, kmeans, dca, nocompress).')
    parser.add_argument('--rungroup', type=str, required=True,
        help='Name of rungroup')
    parser.add_argument('--bitrate', type=int, required=True,
        help='Bitrate.  Note not exact for some methods, but as close as possible.')
    parser.add_argument('--seed', type=int, required=True,
        help='Random seed to use for experiment.')
    parser.add_argument('--debug', action='store_true',
        help='If set to false, can have local git changes when running this.')
    ### Begin uniform quantization hyperparameters
    parser.add_argument('--stoch', action='store_true', 
        help='Specifies whether stochastic quantization should be used.')
    parser.add_argument('--adaptive', action='store_true', 
        help='Specifies whether range for uniform quantization should be optimized.')
    parser.add_argument('--skipquant', action='store_true', 
        help='Specifies whether clipping should be performed without quantization (for ablation).')
    ### End uniform quantization hyperparameters
    ### Begin DCA hyperparameters
    # The number of DCA codebooks 'm' is determined from k and bitrate
    # k is a required argument for DCA training.
    parser.add_argument('--k', type=int, default=-1,
        help='Codebook size for DCA, must be a power of 2.')
    parser.add_argument('--lr', type=float, default=0.0001,
        help='Learning rate for DCA training.')
    parser.add_argument('--batchsize', type=int, default=64,
        help='Batch size for DCA training.')
    parser.add_argument('--temp', type=float, default=1.0,
        help='Temperature parameter for DCA training.')
    parser.add_argument('--gradclip', type=float, default=0.001,
        help='Clipping of gradient norm for DCA training.')
    ### End DCA hyperparameters
    return parser

def init_evaluate_parser():
    """Initialize cmd-line parser for embedding evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaltype', type=str, required=True,
        choices=['QA','intrinsics','synthetics'],
        help='Evaluation type.')
    parser.add_argument('--embedpath', type=str, required=True,
        help='Path to embedding to evaluate.')
    parser.add_argument('--seed', type=int, required=True,
        help='Random seed to use for experiment.')
    parser.add_argument('--epochs', type=int, default=50,
        help='Number of epochs to run for DrQA training.')
    return parser

def init_config(parser, runtype):
    global config
    config = vars(parser.parse_args())
    validate_config(runtype)
    orig_config = config.copy()
    config['runname'] = get_runname(parser, runtype)
    config['datestr'] = get_date_str()
    config['rungroup'] =  '{}-{}'.format(config['datestr'], config['rungroup'])
    config['full-runname'] = get_full_runname(runtype)
    windows_dir = str(pathlib.PurePath(get_windows_home_dir(), 'Babel_Files','smallfry'))
    config['basedir'] = windows_dir if is_windows() else '/proj/smallfry'
    config['rundir'] = get_and_make_run_dir(runtype)    
    init_logging()
    config['githash'], config['gitdiff'] = get_git_hash_and_diff() # might fail
    logging.info('Command line arguments: {}'.format(' '.join(sys.argv[1:])))
    initialize_further(runtype)
    save_dict_as_json(orig_config, get_filename('_orig_config.json'))
    save_current_config()

def save_current_config():
    save_dict_as_json(config, get_filename('_config.json'))

def initialize_further(runtype):
    global config
    if runtype == 'train':
        pass # TODO
    elif runtype == 'compress':
        pass # TODO
    elif runtype == 'evaluate':
        pass # TODO

def validate_config(runtype):
    assert '_' not in config['rungroup'], 'rungroups should not have underscores'
    if runtype == 'train':
        pass # TODO
    elif runtype == 'compress':
        if config['compresstype'] == 'dca':
            assert config['k'] == -1, 'Must specify k for DCA training.'
            assert np.log2(config['k']) == np.ceil(np.log2(config['k'])), \
                'k must be a power of 2.'
        if config['embedtype'] == 'glove400k':
            assert config['embeddim'] in (50,100,200,300)
    elif runtype == 'evaluate':
        pass # TODO

def get_runname(parser, runtype):
    runname = ''
    if runtype == 'train':
        pass # TODO
    elif runtype == 'compress':    
        to_skip = ('embedtype','compresstype','rungroup')
    elif runtype == 'evaluate':
        pass # TODO

    for key,val in non_default_args(parser, config):
        if key not in to_skip:
            runname += '{},{}_'.format(key,val)
    # remove the final '_' from runname
    return runname[0:-1] if runname[-1] == '_' else runname

def get_full_runname(runtype):
    if runtype == 'train':
        pass # TODO
    elif runtype == 'compress':
        return 'embedtype,{}_compresstype,{}_rungroup,{}_{}'.format( 
            config['embedtype'], config['compresstype'], 
            config['rungroup'], config['runname'])
    elif runtype == 'evaluate':
        pass # TODO

def get_and_make_run_dir(runtype):
    rundir = ''
    if runtype == 'train':
        pass # TODO
    elif runtype == 'compress':
        rundir = str(pathlib.PurePath(
            config['basedir'], 'embeddings', config['embedtype'], 
            config['rungroup'], config['runname']))
        ensure_dir(rundir)
    elif runtype == 'evaluate':
        pass # TODO
    return rundir

def init_random_seeds():
    """Initialize random seeds."""
    torch.manual_seed(config['seed'])
    tf.set_random_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    if config['cuda']:
        torch.cuda.manual_seed(config['seed'])

def args_set_at_cmdline():
    args_set = [s[2:] for s in sys.argv[1:] if (len(s) > 2 and s[0:2] == '--')]
    assert len(args_set) > 0, 'There must be a cmdline arg.'
    return args_set

def non_default_args(parser, args):
    non_default = []
    for action in parser._actions:
        default = action.default
        key = action.dest
        if key in args:
            val = args[key]
            if val != default:
                non_default.append((key,val))
    assert len(non_default) > 0, 'There must be a non-default arg.'
    return non_default

def get_git_hash_and_diff():
    git_hash = None
    git_diff = None
    try:
        wd = os.getcwd()
        git_repo_dir = '/proj/mlnlp/avnermay/Babel/Git/smallfry'
        os.chdir(git_repo_dir)
        git_hash = str(check_output(['git','rev-parse','--short','HEAD']).strip())[:9]
        git_diff = str(check_output(['git','diff']).strip())
        if not config['debug']:
            # if not in debug mode, local repo changes are not allowed.
            assert git_diff == '', 'Cannot have any local changes'
        os.chdir(wd)
        logging.info('Git hash {}'.format(git_hash))
        logging.info('Git diff {}'.format(git_diff))
    except FileNotFoundError:
        logging.info('Unable to get git hash.')
    return git_hash, git_diff

def get_date_str():
    return '{:%Y-%m-%d}'.format(datetime.date.today())

def ensure_dir(dir):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

def get_filename(suffix):
    """Return filename which is 'rundir/full-runname + suffix'."""
    return str(pathlib.PurePath(
        config['rundir'], config['full-runname'] + suffix))

def init_logging():
    """Initialize logfile to be used for experiment."""
    log_filename = get_filename('.log')
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(message)s',
                        datefmt='[%m/%d/%Y %H:%M:%S]: ',
                        filemode='w', # this will overwrite existing log file.
                        level=logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)
    logging.info('Begin logging.')

def save_dict_as_json(dict_to_write, path):
    with open(path, 'w') as f: json.dump(dict_to_write, f, indent=2)

def load_dict_from_json(path):
    with open(path) as f: return json.load(f)

def is_windows():
    """Determine if running on windows OS."""
    return os.name == 'nt'

def get_windows_home_dir():
    return (r'C:\Users\Avner' if
            socket.gethostname() == 'Avner-X1Carbon' else
            r'C:\Users\avnermay')

def load_embeddings(path):
    """
    Loads a GloVe embedding at specified path. Returns a vector of strings that 
    represents the vocabulary and a 2-D numpy matrix that is the embeddings. 
    """
    with open(path, 'r') as f:
        lines = f.readlines()
        wordlist = []
        embeddings = []
        for line in lines:
            row = line.strip("\n").split(" ")
            wordlist.append(row.pop(0))
            embeddings.append([float(i) for i in row])
        embeddings = np.array(embeddings)
    assert len(wordlist) == embeddings.shape[0], 'Embedding dim must match wordlist length.'
    return embeddings, wordlist

def save_embeddings(path, embeds, wordlist):
    ''' save embeddings in text file format'''
    with open(path, "w+") as file:
        for i in range(len(wordlist)):
            file.write(wordlist[i] + " ")
            row = embeds[i, :]
            strrow = [str(r) for r in row]
            file.write(" ".join(strrow))
            file.write("\n")
