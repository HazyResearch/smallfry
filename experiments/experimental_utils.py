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
   git_hash = None
   try:
       this_dir = os.path.dirname(os.path.realpath(__file__))
       git_hash = check_output(['cd %s; git' % this_dir,'rev-parse','--short','HEAD']).strip()
       logging.info('Git hash {}'.format(git_hash))
   except FileNotFoundError:
       logging.info('Unable to get git hash.')
   return str(git_hash)

def codes_2_vec(codes, codebook, m ,k ,v,d):
    codes = codes.reshape(int(len(codes)/m),m)
    dcc_mat = np.zeros([v,d])
    for i in range(v):
        for j in range(m):
            dcc_mat[i,:] += codebook[j*k+codes[i,j],:]
    return dcc_mat

def save_dict_as_json(dict_to_write, path):
    with open(path, 'w') as f: json.dump(dict_to_write, f, indent=2)

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

def eval_print(message):
    callername = sys._getframe().f_back.f_code.co_name
    tsstring = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print("%s-%s : %s" % (tsstring, callername, message))
    sys.stdout.flush()

def perform_command_local(command):
    out = check_output(command, stderr=subprocess.STDOUT, shell=True).decode("utf-8") 
    return out     

def get_results_path(embed_path, results_type, results):
    embed_name = os.path.basename(embed_path)
    results_file = '%s_results-%s.json' % (embed_name, results_type)
    return str(pathlib.PurePath(embed_path, results_file))

def get_log_name(name, rungroup):
    return '%s:%s:%s' % (get_date_str(), rungroup, name)

def prep_qsub_log_dir(qsub_log_path_generic, name, rungroup):
    qsub_log_path_specific = str(pathlib.PurePath(qsub_log_path_generic,
                                                 get_log_name(name, rungroup)))
    os.mkdir(qsub_log_path_specific)

def do_results_already_exist(embed_path, results_type, results):
    results_path = get_results_path(embed_path, results_type, results)
    return os.path.isfile(results_path)

def results_to_file(embed_path, results_type, results):
    results_path = get_results_path(embed_path, results_type, results)
    with open(results_path, 'w+') as results_f:
            results_f.write(json.dumps(results)) 

def fetch_embeds_txt_path(embed_path):
    embed_name = os.path.basename(embed_path)
    return str(pathlib.PurePath(embed_path, embed_name+'.txt'))

def fetch_embeds_4_eval(embed_path):
    embed_txt_path = fetch_embeds_txt_path(embed_path)
    embeds, wordlist = load_embeddings(embed_txt_path) #clarify what load_embeddings returns
    assert len(embeds) == len(wordlist), 'Embeddings and wordlist have different lengths in eval.py'
    return embeds, wordlist

def fetch_dim(embed_path):
    embed_name = os.path.basename(embed_path)
    maker_config_path = str(pathlib.PurePath(embed_path, embed_name+'_config.json'))
    maker_config = dict()
    with open(maker_config_path, 'r') as maker_config_f:
        maker_config = json.loads(maker_config_f.read())
    return maker_config['dim']

def fetch_base_embed_path(embed_path):
    embed_name = os.path.basename(embed_path)
    maker_config_path = str(pathlib.PurePath(embed_path, embed_name+'_config.json'))
    maker_config = dict()
    with open(maker_config_path, 'r') as maker_config_f:
        maker_config = json.loads(maker_config_f.read())
    return maker_config['basepath']

def get_drqa_directory():
    return "/proj/smallfry/embeddings_benchmark/DrQA/"

def get_relation_directory():
    return "/proj/smallfry/embeddings_benchmark/tacred-relation/"

def get_sentiment_directory():
    return "/proj/smallfry/embeddings_benchmark/compositional_code_learning/"

def get_base_embed_path_head():
    return '/proj/smallfry/base_embeddings'

def get_base_outputdir():
    return '/proj/smallfry/embeddings'

def get_launch_path():
    return '/proj/smallfry/launches'

def get_qsub_log_path():
    return '/proj/smallfry/qsub_logs'

