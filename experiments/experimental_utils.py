import json
import datetime
import logging
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