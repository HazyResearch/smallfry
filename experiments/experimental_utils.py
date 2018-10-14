import json
import datetime
import logging
import os
import pathlib
import subprocess
import sys
import time
import numpy as np
from subprocess import check_output
from smallfry.smallfry import Smallfry
from smallfry.utils import *

def get_git_hash():
    ''' returns githash of current directory. NOTE: not safe before init_logging'''
    git_hash = None
    try:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        git_hash = check_output(['git','rev-parse','--short','HEAD'],cwd=this_dir).strip()
        logging.info('Git hash {}'.format(git_hash))
    except FileNotFoundError:
        logging.info('Unable to get git hash.')
    return str(git_hash)

def codes_2_vec(codes, codebook, m ,k ,v,d):
    ''' reshapes inflates DCA output embeddings -- assumes input is properly formatted and flattened'''
    codes = codes.reshape(int(len(codes)/m),m)
    dcc_mat = np.zeros([v,d])
    for i in range(v):
        for j in range(m):
            dcc_mat[i,:] += codebook[j*k+codes[i,j],:]
    return dcc_mat

def compute_m_dca(k, v, d, br):
    return int(np.round(0.125*br*v*d/(0.125*v*np.log2(k) + 4*d*k)))

def save_dict_as_json(dict_to_write, path):
    ''' pydict to json --> this method is fairly pointless and not really used '''
    with open(path, 'w') as f: json.dump(dict_to_write, f, indent=2)

def get_date_str():
    ''' gets datetime '''
    return '{:%Y-%m-%d}'.format(datetime.date.today())

def to_file_np(path, embeds):
    ''' saves np file '''
    np.save(path, embeds)

def to_file_txt(path, wordlist, embeds):
    ''' save embeddings in text file format'''
    with open(path, "w+") as file:
        for i in range(len(wordlist)):
            file.write(wordlist[i] + " ")
            row = embeds[i, :]
            strrow = [str(r) for r in row]
            file.write(" ".join(strrow))
            file.write("\n")

def init_logging(log_filename):
    """Initialize logfile to be used for experiment."""
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(message)s',
                        datefmt='[%m/%d/%Y %H:%M:%S]: ',
                        filemode='w', # this will overwrite existing log file.
                        level=logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)
    logging.info('Begin logging.')

def eval_print(message):
    ''' fancy print out stuff -- author: MAXLAM'''
    callername = sys._getframe().f_back.f_code.co_name
    tsstring = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print("%s-%s : %s" % (tsstring, callername, message))
    sys.stdout.flush()

def perform_command_local(command):
    ''' performs a command -- author: MAXLAM'''
    out = check_output(command, stderr=subprocess.STDOUT, shell=True).decode('utf-8') 
    return out     

def get_results_path(embed_path, results_type):
    ''' gets path to a certain results-type given embeddings path'''
    embed_name = os.path.basename(embed_path)
    results_file = '%s_results-%s.json' % (embed_name, results_type)
    return str(pathlib.PurePath(embed_path, results_file))

def get_log_name(name, rungroup):
    ''' constructs standard launch log filename format'''
    return '%s:%s:%s' % (get_date_str(), rungroup, name)

def get_qsub_preamble():
    return "qsub -V -b y -wd"

def prep_qsub_log_dir(qsub_log_path_generic, name, rungroup):
    ''' makes log dir for qsubs'''
    qsub_log_path_specific = str(pathlib.PurePath(qsub_log_path_generic,
                                                 get_log_name(name, rungroup)))
    os.mkdir(qsub_log_path_specific)
    return qsub_log_path_specific

def launch_config(config, action, action_path, qsub=False):
    s = ''
    assert action in ['eval','gen','maker'], f"Invalid action in config launcher:{action}"
    qsub_log_path = str(pathlib.PurePath(get_qsub_log_path(), action))
    python_action_cmd = f"python {action_path} "
    flags = [python_action_cmd]
    for key in config.keys():
        flags.append(f"--{key} {config[key]}")
    s = " ".join(flags)
    s = f"{get_qsub_preamble()} {qsub_log_path} {s}" if qsub else s
    return s

def do_results_already_exist(embed_path, results_type):
    ''' boolean function -- default behavior is to fail when results already exist'''
    results_path = get_results_path(embed_path, results_type)
    return os.path.isfile(results_path)

def results_to_file(embed_path, results_type, results):
    ''' writes json results to file'''
    results_path = get_results_path(embed_path, results_type)
    with open(results_path, 'w+') as results_f:
            results_f.write(json.dumps(results)) 

def fetch_embeds_txt_path(embed_path):
    ''' returns path to text embeddings representation given embeddings dir path'''
    embed_name = os.path.basename(embed_path)
    return str(pathlib.PurePath(embed_path, embed_name+'.txt'))

def fetch_embeds_npy_path(embed_path):
    ''' returns path to npy embeddings representation given embeddings dir path'''
    embed_name = os.path.basename(embed_path)
    return str(pathlib.PurePath(embed_path, embed_name+'.npy')) 

def fetch_embeds_config_path(embed_path):
    ''' returns path to npy embeddings representation given embeddings dir path'''
    embed_name = os.path.basename(embed_path)
    return str(pathlib.PurePath(embed_path, embed_name+'_config.json'))

def fetch_embeds_makelog_path(embed_path):
    ''' returns path to npy embeddings representation given embeddings dir path'''
    embed_name = os.path.basename(embed_path)
    return str(pathlib.PurePath(embed_path, embed_name+'_maker.log'))

def fetch_embeds_4_eval(embed_path):
    ''' returns numpy embeddings and wordlist using smallfry load embeddings'''
    embed_txt_path = fetch_embeds_txt_path(embed_path)
    embeds, wordlist = load_embeddings(embed_txt_path) #clarify what load_embeddings returns
    assert len(embeds) == len(wordlist), 'Embeddings and wordlist have different lengths in eval.py'
    return embeds, wordlist

def fetch_maker_config(embed_path):
    '''reads and returns the maker config'''
    embed_name = os.path.basename(embed_path)
    maker_config_path = str(pathlib.PurePath(embed_path, embed_name+'_config.json'))
    maker_config = dict()
    with open(maker_config_path, 'r') as maker_config_f:
        maker_config = json.loads(maker_config_f.read())
    return maker_config

def fetch_dim(embed_path): #TODO: deprecate this method in favor of above
    '''reads embedding dimension from maker config '''
    return fetch_maker_config(embed_path)['dim']

def fetch_base_embed_path(embed_path):
    ''' reads path to base embeddings from maker config'''
    embed_name = os.path.basename(embed_path)
    maker_config_path = str(pathlib.PurePath(embed_path, embed_name+'_config.json'))
    maker_config = dict()
    with open(maker_config_path, 'r') as maker_config_f:
        maker_config = json.loads(maker_config_f.read())
    return maker_config['basepath']

def get_environment():
    ''' Use this routine to determine your compute environment'''
    host = perform_command_local('hostname')
    if 'smallfry' in host: # In Avner's smallfry AWS image
        return 'AWS'
    elif 'dawn' in host: #On the DAWN cluster
        return 'DAWN'
    elif 'ThinkPad-X270' in host: #on Tony's laptop
        return 'TONY'

def whoami():
    '''Wraps bash whoami'''
    return perform_command_local('whoami')[:-1] # last char is newline, so drop it

'''HARDCODED PATHS BELOW'''

def get_base_directory():
    path = '/proj/smallfry'
    if get_environment() == 'DAWN':
        path = '/lfs/1/%s%s' % (whoami(),path) 
    return path

def get_drqa_directory():
    return str(pathlib.PurePath(get_base_directory(), "embeddings_benchmark/DrQA/"))

def get_relation_directory():
    return str(pathlib.PurePath(get_base_directory(),"embeddings_benchmark/tacred-relation/"))

def get_senwu_sentiment_directory():
    return str(pathlib.PurePath(get_base_directory(),"embeddings_benchmark/sentence_classification"))

def get_harvardnlp_sentiment_data_directory():
    return str(pathlib.PurePath(get_base_directory(),"embeddings_benchmark/sent-conv-torch/data"))

def get_senwu_sentiment_out_directory():
    return str(pathlib.PurePath(get_base_directory(), "senwu_sentiment_out"))

def get_sentiment_directory():
    return str(pathlib.PurePath(get_base_directory(),"embeddings_benchmark/compositional_code_learning/"))

def get_base_embed_path_head():
    return str(pathlib.PurePath(get_base_directory(),'base_embeddings'))

def get_base_outputdir():
    return str(pathlib.PurePath(get_base_directory(),'embeddings'))

def get_launch_path():
    return str(pathlib.PurePath(get_base_directory(),'launches'))

def get_qsub_log_path():
    return str(pathlib.PurePath(get_base_directory(),'qsub_logs'))

def get_plots_path():
    return str(pathlib.PurePath(get_base_directory(),'plots'))

def get_corpus_path():
    return str(pathlib.PurePath(get_base_directory(),'corpus'))

def get_glove_generator_path():
    return str(pathlib.PurePath(get_base_directory(),'smallfry/experiments/generation/GloVe'))
