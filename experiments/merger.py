
import glob
import pathlib
import json

def merger(basedir,query):
    #USER NOTE: query matches for RUNGROUPS!
    qry = pathlib.PurePath(basedir,query)
    d_list = []
    for e in glob.glob(str(qry)): #BUG ALERT
        qry_dict = pathlib.PurePath(qry,'*.json')
        e_dict = {}
        for file in glob.glob(str(qry_dict)):
            d = json.loads(file)
            for k in d.key():
                assert k not in e_dict, "duplicated fields in json dicts"
                e_dict[k] = d[k]
        d_list.append(e_dict)

    return d_list    


def get_seeds(d_list, base, vocab, method):
    seeds = []
    for d in d_list:
        if d['base'] == base and d['vocab'] == vocab and d['method'] == method:
            seeds.append(d['seed'])
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

