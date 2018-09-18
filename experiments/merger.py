
import glob
import pathlib
import json
import os

def merger(basedir,query):
    '''
    Simple aggregation routine.
    basedir -- the embeddings base directory
    query -- a Unix-style (supports wildcards) query for EMBEDDINGS
    query example: my-rungroup/my-embeds* -- captures all embeddings in 'my-rungroup' that start with 'my-embeds'
    '''
    #USER NOTE: query matches for RUNGROUPS!
    qry = str(pathlib.PurePath(basedir,query))
    d_list = []
    for emb in glob.glob(qry): #BUG ALERT
        emb_data_qry = str(pathlib.PurePath(emb,'*.json'))
        e_dict = {}
        for data in glob.glob(emb_data_qry):
            with open(data,'r') as data_json:
                d = json.loads(data_json.read())
            for k in d.keys():
                assert k not in e_dict, "duplicated fields in json dicts"
                e_dict[k] = d[k]
        d_list.append(e_dict)

    return d_list    


def get_seeds(d_list, base, vocab, method):
    seeds = []
    for d in d_list:
        if d['base'] == base and d['vocab'] == vocab and d['method'] == method:
            seeds.append(d['seed'])
    seeds = list(set(seeds))
    seeds.sort()
    return seeds


def get_data(d_list, base, vocab, method, seeds, x, y):
    seeds.sort()
    res = []
    for i in range(len(seeds)):
        res.append([])
        for d in d_list:
            if d['base'] == base and d['vocab'] == vocab and d['method'] == method and d['seed'] == seeds[i]:
                res[i].append((d[x],d[y]))
            res[i].sort()
        return res

