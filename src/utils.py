import json
import datetime
import logging
import os
import pathlib
import subprocess
import sys
import time
import random
import argparse
import glob
import torch
import numpy as np
from subprocess import check_output

config = {}

def init_train():
    parser = init_train_parser()
    init_train_config(parser)
    init_random_seeds()

def init_compress():
    parser = init_compress_parser()
    init_compress_config(parser)
    init_random_seeds()

def init_evalaute():
    parser = init_evaluate_parser()
    init_evaluate_config(parser)
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
    parser.add_argument('--compresstype', type=str, required=True,
        choices=['uniform','kmeans','dca','nocompress'],
        help='Name of compression method to use (uniform, kmeans, dca, nocompress).')
    parser.add_argument('--rungroup', type=str, required=True,
        help='Name of rungroup')
    parser.add_argument('--ibr', type=int, required=True,
        help='Intended bitrate.')
    parser.add_argument('--seed', type=int, required=True,
        help='Random seed to use for experiment.')
    # Begin DCA hyperparameters
    parser.add_argument('--k', type=int, default=-1,
        help='Codebook size for DCA, must be a power of 2.')
    parser.add_argument('--debug', action='store_true',
        help='If set to false, can have local git changes when running this.')
    # The number of DCA codebooks 'm' is determined from k and ibr.
    # k is a required argument for DCA training.
    parser.add_argument('--temp', type=float, default=1.0,
        help='Temperature parameter for DCA training.')
    parser.add_argument('--batchsize', type=int, default=64,
        help='Batch size for DCA training.')
    parser.add_argument('--gradclip', type=float, default=0.001,
        help='Clipping of gradient norm for DCA training.')
    parser.add_argument('--lr', type=float, default=0.0001,
        help='Learning rate for DCA training.')
    # End DCA hyperparameters
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

def init_compress_config(parser):
    global config
    config = vars(parser.parse_args())
    validate_compress_config(config)
    orig_config = config.copy()
    config['runname'] = get_compress_runname(parser)
    config['datestr'] = get_date_str()
    config['rungroup'] =  '{}-{}'.format(config['datestr'], config['rungroup'])
    config['full-runname'] = \
        'embedtype,{}_compresstype,{}_rungroup,{}_{}'.format( 
        config['embedtype'], config['compresstype'], 
        config['rungroup'], config['runname'])
    windows_dir = str(pathlib.PurePath(get_windows_home_dir(), 'Babel_Files'))
    config['basedir'] = (windows_dir if is_windows() else 
                      '/proj/smallfry')
    config['rundir'] = str(pathlib.PurePath(
        config['basedir'], 'embeddings', config['embedtype'], 
        config['rungroup'], config['runname']))
    ensure_dir(config['rundir'])
    init_logging()
    config['githash'], config['gitdiff'] = get_git_hash_and_diff() # might fail
    logging.info('Command line arguments: {}'.format(' '.join(sys.argv[1:])))
    save_dict_as_json(config, get_filename('_config.json'))
    save_dict_as_json(orig_config, get_filename('_orig_config.json'))

def validate_compress_config(cfg):
    if cfg['compresstype'] == 'dca':
        assert config['k'] == -1, 'Must specify k for DCA training.'
        assert np.log2(config['k']) == np.ceil(np.log2(config['k'])), \
               'k must be a power of 2.'

def get_compress_runname(parser):
    runname = ''
    for key,val in non_default_args(parser,config):
        if key not in ('embedtype','compresstype','rungroup'):
            runname += '{},{}_'.format(key,val)
    # remove the final '_' from runname
    return runname[0:-1]

def init_random_seeds():
    """Initialize random seeds."""
    torch.manual_seed(config['seed'])
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

#==================================================================
# TONY'S CODE ====================================================
#==================================================================
# def get_git_hash():
#     ''' returns githash of current directory. NOTE: not safe before init_logging'''
#     git_hash = None
#     try:
#         this_dir = os.path.dirname(os.path.realpath(__file__))
#         git_hash = check_output(['git','rev-parse','--short','HEAD'],cwd=this_dir).strip()
#         logging.info('Git hash {}'.format(git_hash))
#     except FileNotFoundError:
#         logging.info('Unable to get git hash.')
#     return str(git_hash)

# def codes_2_vec(codes, codebook, m ,k ,v,d):
#     ''' reshapes inflates DCA output embeddings -- assumes input is properly formatted and flattened'''
#     codes = codes.reshape(int(len(codes)/m),m)
#     dcc_mat = np.zeros([v,d])
#     for i in range(v):
#         for j in range(m):
#             dcc_mat[i,:] += codebook[j*k+codes[i,j],:]
#     return dcc_mat

# def compute_m_dca(k, v, d, br):
#     return int(np.round(0.125*br*v*d/(0.125*v*np.log2(k) + 4*d*k)))

# def set_seeds(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)

# def save_dict_as_json(dict_to_write, path):
#     ''' pydict to json --> this method is fairly pointless and not really used '''
#     with open(path, 'w') as f: json.dump(dict_to_write, f, indent=2)

# def get_date_str():
#     ''' gets datetime '''
#     return '{:%Y-%m-%d}'.format(datetime.date.today())

# def to_file_np(path, embeds):
#     ''' saves np file '''
#     np.save(path, embeds)

# def to_file_txt(path, wordlist, embeds):
#     ''' save embeddings in text file format'''
#     with open(path, "w+") as file:
#         for i in range(len(wordlist)):
#             file.write(wordlist[i] + " ")
#             row = embeds[i, :]
#             strrow = [str(r) for r in row]
#             file.write(" ".join(strrow))
#             file.write("\n")

# def init_logging(log_filename):
#     """Initialize logfile to be used for experiment."""
#     logging.basicConfig(filename=log_filename,
#                         format='%(asctime)s %(message)s',
#                         datefmt='[%m/%d/%Y %H:%M:%S]: ',
#                         filemode='w', # this will overwrite existing log file.
#                         level=logging.DEBUG)
#     console = logging.StreamHandler(sys.stdout)
#     console.setLevel(logging.DEBUG)
#     logging.getLogger('').addHandler(console)
#     logging.info('Begin logging.')

# def eval_print(message):
#     ''' fancy print out stuff -- author: MAXLAM'''
#     callername = sys._getframe().f_back.f_code.co_name
#     tsstring = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#     print("%s-%s : %s" % (tsstring, callername, message))
#     sys.stdout.flush()

# def perform_command_local(command):
#     ''' performs a command -- author: MAXLAM'''
#     out = check_output(command, stderr=subprocess.STDOUT, shell=True).decode('utf-8') 
#     return out     

# def get_results_path(embed_path, results_type):
#     ''' gets path to a certain results-type given embeddings path'''
#     embed_name = os.path.basename(embed_path)
#     results_file = '%s_results-%s.json' % (embed_name, results_type)
#     return str(pathlib.PurePath(embed_path, results_file))

# def get_log_name(name, rungroup):
#     ''' constructs standard launch log filename format'''
#     return '%s:%s:%s' % (get_date_str(), rungroup, name)

# def get_qsub_preamble():
#     return "qsub -V -b y -wd"

# def prep_qsub_log_dir(qsub_log_path_generic, name, rungroup):
#     ''' makes log dir for qsubs'''
#     qsub_log_path_specific = str(pathlib.PurePath(qsub_log_path_generic,
#                                                  get_log_name(name, rungroup)))
#     os.mkdir(qsub_log_path_specific)
#     return qsub_log_path_specific

# def launch_config(config, action, action_path, qsub=False):
#     s = ''
#     assert action in ['eval','gen','maker'], f"Invalid action in config launcher:{action}"
#     qsub_log_path = str(pathlib.PurePath(get_qsub_log_path(), action))
#     python_action_cmd = f"python {action_path} "
#     flags = [python_action_cmd]
#     for key in config.keys():
#         flags.append(f"--{key} {config[key]}")
#     s = " ".join(flags)
#     s = f"{get_qsub_preamble()} {qsub_log_path} {s}" if qsub else s
#     return s 
 
# def get_all_embs_in_rg(rungroup):
#     rungroup_qry = f"{get_base_outputdir()}/*"
#     rungroup_found = False
#     embs = []
#     for rg in glob.glob(rungroup_qry):
#         if os.path.basename(rg) == rungroup:    
#             rungroup_found = True
#             rungroup_wildcard = rg +'/*'
#             for emb in glob.glob(rungroup_wildcard):
#                 embs.append(emb)
#     assert rungroup_found, f"rungroup requested {rungroup} not found"
#     return embs

# def do_results_already_exist(embed_path, results_type):
#     ''' boolean function -- default behavior is to fail when results already exist'''
#     results_path = get_results_path(embed_path, results_type)
#     print(results_path)
#     return os.path.isfile(results_path)

# def results_to_file(embed_path, results_type, results):
#     ''' writes json results to file'''
#     results_path = get_results_path(embed_path, results_type)
#     with open(results_path, 'w+') as results_f:
#             results_f.write(json.dumps(results)) 

# def fetch_embeds_txt_path(embed_path):
#     ''' returns path to text embeddings representation given embeddings dir path'''
#     embed_name = os.path.basename(embed_path)
#     return str(pathlib.PurePath(embed_path, embed_name+'.txt'))

# def fetch_embeds_npy_path(embed_path):
#     ''' returns path to npy embeddings representation given embeddings dir path'''
#     embed_name = os.path.basename(embed_path)
#     return str(pathlib.PurePath(embed_path, embed_name+'.npy')) 

# def fetch_embeds_config_path(embed_path):
#     ''' returns path to npy embeddings representation given embeddings dir path'''
#     embed_name = os.path.basename(embed_path)
#     return str(pathlib.PurePath(embed_path, embed_name+'_config.json'))

# def fetch_embeds_makelog_path(embed_path):
#     ''' returns path to npy embeddings representation given embeddings dir path'''
#     embed_name = os.path.basename(embed_path)
#     return str(pathlib.PurePath(embed_path, embed_name+'_maker.log'))

# def fetch_embeds_4_eval(embed_path):
#     ''' returns numpy embeddings and wordlist using smallfry load embeddings'''
#     embed_txt_path = fetch_embeds_txt_path(embed_path)
#     embeds, wordlist = load_embeddings(embed_txt_path) #clarify what load_embeddings returns
#     assert len(embeds) == len(wordlist), 'Embeddings and wordlist have different lengths in eval.py'
#     return embeds, wordlist

# def fetch_maker_config(embed_path):
#     '''reads and returns the maker config'''
#     embed_name = os.path.basename(embed_path)
#     maker_config_path = str(pathlib.PurePath(embed_path, embed_name+'_config.json'))
#     maker_config = dict()
#     with open(maker_config_path, 'r') as maker_config_f:
#         maker_config = json.loads(maker_config_f.read())
#     return maker_config

# def fetch_dim(embed_path): #TODO: deprecate this method in favor of above
#     '''reads embedding dimension from maker config '''
#     return fetch_maker_config(embed_path)['dim']

# def fetch_base_embed_path(embed_path):
#     ''' reads path to base embeddings from maker config'''
#     embed_name = os.path.basename(embed_path)
#     maker_config_path = str(pathlib.PurePath(embed_path, embed_name+'_config.json'))
#     maker_config = dict()
#     with open(maker_config_path, 'r') as maker_config_f:
#         maker_config = json.loads(maker_config_f.read())
#     return maker_config['basepath']

# def get_environment():
#     ''' Use this routine to determine your compute environment'''
#     host = perform_command_local('hostname')
#     if 'smallfry' in host: # In Avner's smallfry AWS image
#         return 'AWS'
#     elif 'dawn' in host: #On the DAWN cluster
#         return 'DAWN'
#     elif 'ThinkPad-X270' in host: #on Tony's laptop
#         return 'TONY'

# def whoami():
#     '''Wraps bash whoami'''
#     return perform_command_local('whoami')[:-1] # last char is newline, so drop it

# '''HARDCODED PATHS BELOW'''

# def get_base_directory():
#     path = '/proj/smallfry'
#     if get_environment() == 'DAWN':
#         path = '/lfs/1/%s%s' % (whoami(),path) 
#     return path

# def get_drqa_directory():
#     return str(pathlib.PurePath(get_base_directory(), "embeddings_benchmark/DrQA/"))

# def get_relation_directory():
#     return str(pathlib.PurePath(get_base_directory(),"embeddings_benchmark/tacred-relation/"))

# def get_senwu_sentiment_directory():
#     return str(pathlib.PurePath(get_base_directory(),"embeddings_benchmark/sentence_classification"))

# def get_harvardnlp_sentiment_data_directory():
#     return str(pathlib.PurePath(get_base_directory(),"embeddings_benchmark/sent-conv-torch/data"))

# def get_senwu_sentiment_out_directory():
#     return str(pathlib.PurePath(get_base_directory(), "senwu_sentiment_out"))

# def get_sentiment_directory():
#     return str(pathlib.PurePath(get_base_directory(),"embeddings_benchmark/compositional_code_learning/"))

# def get_base_embed_path_head():
#     return str(pathlib.PurePath(get_base_directory(),'base_embeddings'))

# def get_base_outputdir():
#     return str(pathlib.PurePath(get_base_directory(),'embeddings'))

# def get_launch_path():
#     return str(pathlib.PurePath(get_base_directory(),'launches'))

# def get_qsub_log_path():
#     return str(pathlib.PurePath(get_base_directory(),'qsub_logs'))

# def get_plots_path():
#     return str(pathlib.PurePath(get_base_directory(),'plots'))

# def get_corpus_path():
#     return str(pathlib.PurePath(get_base_directory(),'corpus'))

# def get_agg_results_path():
#     return str(pathlib.PurePath(get_base_directory(),'results'))

# def get_glove_generator_path():
#     return str(pathlib.PurePath(get_base_directory(),'smallfry/experiments/generation/GloVe'))

# def load_embeddings(embeds_txt_filepath):
#     """
#     Loads a GloVe embedding at 'filename'. Returns a vector of strings that 
#     represents the vocabulary and a 2-D numpy matrix that is the embeddings. 
#     """
#     with open(embeds_txt_filepath, 'r') as embeds_file:
#         lines = embeds_file.readlines()
#         wordlist = []
#         embeddings = []
#         for line in lines:
#             row = line.strip("\n").split(" ")
#             wordlist.append(row.pop(0))
#             embeddings.append([float(i) for i in row])
#         embeddings = np.array(embeddings)
#     return embeddings, wordlist

# # LOAD EMBEDDINGS WITH WORD TRIE
# # def load_embeddings_trie(embeds_txt_filepath):
# #     """
# #     Loads a GloVe embedding at 'filename'. Returns a vector of strings that 
# #     represents the vocabulary and a 2-D numpy matrix that is the embeddings. 
# #     """
# #     with open(embeds_txt_filepath, 'r') as embeds_file:
# #         lines = embeds_file.readlines()
# #         wordlist = []
# #         embeddings = []
# #         for line in lines:
# #             row = line.strip("\n").split(" ")
# #             wordlist.append(row.pop(0))
# #             embeddings.append([float(i) for i in row])
# #         embeddings = np.array(embeddings)
# #         wordtrie = marisa.Trie(wordlist)
# #         trie_order_embeds = np.zeros(embeddings.shape)
# #         for i in range(len(wordlist)):
# #             i_prime = wordtrie[wordlist[i]]
# #             trie_order_embeds[i_prime,:] = embeddings[i,:]
# #     return wordtrie, trie_order_embeds
