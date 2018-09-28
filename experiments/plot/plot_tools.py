
import glob
import pathlib
import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) #FOR LOCAL IMPORTS
from experimental_utils import *


def agg(query, basedir=get_base_outputdir(), expected_num_res=None):
    '''
    Simple aggregation routine.
    basedir -- the embeddings base directory
    query -- a Unix-style (supports wildcards) query for EMBEDDINGS
    query example: my-rungroup/my-embeds* -- captures all embeddings in 'my-rungroup' that start with 'my-embeds'
    '''
    qry = str(pathlib.PurePath(basedir,query))
    d_list = []
    for emb in glob.glob(qry):
        emb_data_qry = str(pathlib.PurePath(emb,'*.json'))
        e_dict = {}
        for data in glob.glob(emb_data_qry):
            with open(data,'r') as data_json:
                d = json.loads(data_json.read())
            for k in d.keys():
                assert k not in e_dict, "duplicated fields in json dicts"
                e_dict[k] = d[k]
        d_list.append(e_dict)
    if expected_num_res != None:
        assert expected_num_res == len(d_list), "The number of results found does not match up the expected number of results"
    return d_list

def get_all_seeds(d_list, base, vocab, method):
    seeds = []
    for d in d_list:
        if d['base'] == base and d['vocab'] == vocab and d['method'] == method:
            seeds.append(d['seed'])
    seeds = list(set(seeds))
    seeds.sort()
    return seeds

def get_all_data(d_list, base, vocab, method, x, y):
    res = dict()
    for d in d_list:
        if d['base'] == base and d['vocab'] == vocab and d['method'] == method and x in d.keys() and y in d.keys():
            if d[x] in res.keys():
                res[d[x]].append(d[y])
            else:
                res [d[x]] = [d[y]]
    return res

def get_data(d_list, base, vocab, method, seeds, x, y):
    seeds.sort()
    res = []
    for i in range(len(seeds)):
        res.append([])
        for d in d_list:
            if d['base'] == base and d['vocab'] == vocab and d['method'] == method and d['seed'] == seeds[i]:
                print(seeds[i])
                print(base)
                print(vocab)
                print(method)
                print(d['ibr'])
                res[i].append((d[x],d[y]))
        res[i].sort()
    return res

def data_formatter(data):
    data_prime = sorted(data, key=lambda tup: tup[0])
    data_x = [x[0] for x in data_prime]
    data_y = [y[1] for y in data_prime]
    return data_x, data_y

def compute_avg(data,compute_var=False,compute_minmax=False):
    data_d = dict()
    count_d = dict()
    store_d = dict()
    for i in range(len(data)):
        for pair in data[i]:
            if pair[0] in count_d.keys(): 
                count_d[pair[0]] += 1
                store_d[pair[0]].append(pair[1])
                data_d[pair[0]] += pair[1]
            else: 
                count_d[pair[0]] = 1
                store_d[pair[0]] = [pair[1]]
                data_d[pair[0]] = pair[1]
    for x in data_d.keys():
        data_d[x] = data_d[x]/count_d[x]
    data_x = sorted(list(data_d))
    data_y = list()
    for i in range(len(data_x)):
        data_y.append(data_d[data_x[i]])
    #if compute_var:

    #rtn = (data_x, data_y)
    #if compute_var:
       # data_var = 
        #rtn = rtn + ()

    return data_x, data_y

def compute_avg_variable_len(data):
    data_x = sorted(list(data.keys()))
    data_y = list()
    for i in range(len(data_x)):
        data_y.append(sum(data[data_x[i]])/len(data[data_x[i]]))
    return data_x, data_y


def compute_std_variable_len(data):
    data_x = sorted(list(data.keys()))
    data_y = list()
    data_y_low = list()
    data_y_high = list()
    for i in range(len(data_x)):
        print(data_x[i])
        print(data[data_x[i]])
        data_y_low.append( min(data[data_x[i]])  )
        data_y_high.append( max(data[data_x[i]]) )
    return data_y_low, data_y_high

def nice_names_lookup(ugly_name):
    ugly_2_nice = dict()
    ugly_2_nice['max-f1'] = 'QA F1 Score'
    ugly_2_nice['semantic-dist'] = 'Average Cosine Distance'
    ugly_2_nice['analogy-avg-score'] = 'Aggregate Word Analogy Score'
    ugly_2_nice['similarity-avg-score'] = 'Aggregate Word Sim. Spearman Correlation'
    ugly_2_nice['embed-fro-dist'] = 'Frobenius Distance'
    ugly_2_nice['bitrate'] = 'Bitrate'
    ugly_2_nice['maketime-secs'] = 'Compression Runtime (secs)'
    ugly_2_nice['glove'] = 'GloVe'
    ugly_2_nice['fasttext'] = 'FastText'
    if ugly_name in ugly_2_nice.keys():
        return ugly_2_nice[ugly_name]
    else:
        return ugly_name
