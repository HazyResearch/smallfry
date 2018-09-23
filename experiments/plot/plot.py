
import glob
import pathlib
import json
import os


def agg(basedir,query):
    '''
    Simple aggregation routine.
    basedir -- the embeddings base directory
    query -- a Unix-style (supports wildcards) query for EMBEDDINGS
    query example: my-rungroup/my-embeds* -- captures all embeddings in 'my-rungroup' that start with 'my-embeds'
    '''
    #USER NOTE: query matches for RUNGROUPS!
    qry = str(pathlib.PurePath(basedir,query))
    print(qry)
    d_list = []
    for emb in glob.glob(qry):
        print(emb)
        emb_data_qry = str(pathlib.PurePath(emb,'*.json'))
        print(emb_data_qry)
        e_dict = {}
        for data in glob.glob(emb_data_qry):
            print(data)
            with open(data,'r') as data_json:
                d = json.loads(data_json.read())
            for k in d.keys():
                assert k not in e_dict, "duplicated fields in json dicts"
                e_dict[k] = d[k]
        d_list.append(e_dict)
    return d_list




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

def data_formatter(data):
    data_prime = sorted(data, key=lambda tup: tup[0])
    data_x = [x[0] for x in data_prime]
    data_y = [y[1] for y in data_prime]
    return data_x, data_y

def compute_avg(data):
    data_d = dict()
    count_d = dict()
    for i in range(len(data)):
        for pair in data[i]:
            if pair[0] in count_d.keys(): 
                count_d[pair[0]] += 1
            else: 
                count_d[pair[0]] = 1
            if pair[0] in data_d.keys():
                data_d[pair[0]] += pair[1]
            else:
                data_d[pair[0]] = pair[1]
    for x in data_d.keys():
        data_d[x] = data_d[x]/count_d[x]
    data_x = sorted(list(data_d))
    data_y = list()
    for i in range(len(data_x)):
        data_y.append(data_d[data_x[i]])
    return data_x, data_y

def get_dca_params(results, bitrates, base):
    res = results
    br_2_mks = dict()
    for br in bitrates:
        br_2_mks[br] = []
        for r in res:
            if r == {} or r['base'] != base: continue
            if r['method'] == 'dca' and abs(r['bitrate'] - br) < 0.15*br:
                br_2_mks[br].append((r['m'],r['k'],r['embed-fro-dist']))
                br_2_mks[br].sort(key=lambda x:x[1])
    return br_2_mks

def get_dca_best_params(results, bitrates, base):
    res = results
    br_2_mk = dict()
    for br in bitrates:
        lowest_mdd = 9999999
        best_res = None
        for r in res:
            if r == {} or r['base'] != base: continue
            if r['method'] == 'dca' and abs(r['bitrate'] - br) < 0.15*br:
                if lowest_mdd > r['embed-fro-dist']:
                    lowest_mdd = r['embed-fro-dist']
                    best_res = r
        br_2_mk[br] = (best_res['m'], best_res['k'])
    return br_2_mk
