import json
import datetime
import logging
import os
import pathlib
import numpy as np
from subprocess import check_output

def get_git_hash():
   git_hash = None
   try:
       git_hash = check_output(['git','rev-parse','--short','HEAD']).strip()
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


def results_to_file(embed_path, results_type, results):
    embed_name = os.path.basename(embed_path)
    results_file = '%s_results-%s.json' % (embed_name, results_type)
    results_path = str(pathlib.PurePath(embed_path, results_file))
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