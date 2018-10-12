
import glob
import pathlib
import json
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from smallfry.utils import load_embeddings
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
        #print(emb_data_qry)
        e_dict = {}
        for data in glob.glob(emb_data_qry):
            with open(data,'r') as data_json:
                d = json.loads(data_json.read())
            for k in d.keys():
                assert k not in e_dict, "duplicated fields in json dicts"
                e_dict[k] = d[k]
        d_list.append(e_dict)
        #print(len(d_list))
    if expected_num_res != None:
        assert expected_num_res == len(d_list),\
        "The number of results found (%s) does not match up the expected number of results (%s). Query: %s" \
        % (len(d_list),expected_num_res,qry)
    return d_list

def get_all_seeds(d_list, base, vocab, method):
    '''
    Extracts all the seeds present for a given base embeddings, vocab, and method
    '''
    seeds = []
    for d in d_list:
        if d['base'] == base and d['vocab'] == vocab and d['method'] == method:
            seeds.append(d['seed'])
    seeds = list(set(seeds))
    seeds.sort()
    return seeds

def get_all_data(d_list, base, vocab, method, x, y):
    '''
    Returns a dictionary such that:
    - key is x
    - values is a list of al y's (for different seeds) matching the query -- no particular order
    '''
    res = dict()
    for d in d_list:
        print(d['base'])
        print(d['method'])
        print(d['vocab'])
        if d['base'] == base and d['vocab'] == vocab and d['method'] == method and x in d.keys() and y in d.keys():
            if d[x] in res.keys():
                res[d[x]].append(d[y])
            else:
                res[d[x]] = [d[y]]
    return res

def get_data_by_seeds(d_list, base, vocab, method, seeds, x, y):
    '''
    Returns a table such that: 
     - each row represents a single random seed, in order of lowest seed to highest
     - each row is a list of tuples of (x,y) pairs each at that row's seed
    '''
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
    '''
    Formats data for easier plotting
    Input: list of list of tuples style
    Output: Parallel lists
    '''
    data_prime = sorted(data, key=lambda tup: tup[0])
    data_x = [x[0] for x in data_prime]
    data_y = [y[1] for y in data_prime]
    return data_x, data_y

def compute_avg(data):
    '''
    Input: data in the format returned by get_all_data
    '''
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
    return data_x, data_y

def compute_avg_variable_len(data):
    data_x = sorted(list(data.keys()))
    data_y = list()
    for i in range(len(data_x)):
        data_y.append(sum(data[data_x[i]])/len(data[data_x[i]]))
    return data_x, data_y

def compute_min_max_variable_len(data):
    data_x = sorted(list(data.keys()))
    data_y = list()
    data_y_low = list()
    data_y_high = list()
    for i in range(len(data_x)):
        data_y_low.append( min(data[data_x[i]])  )
        data_y_high.append( max(data[data_x[i]]) )
    return data_y_low, data_y_high

def get_dca_params(results, bitrates, base):
    '''A simple query routine for getting the results of dca parameter tune'''
    res = results
    br_2_mks = dict()
    for br in bitrates:
        br_2_mks[br] = []
        for r in res:
            if r == {} or r['base'] != base: continue
            if r['method'] == 'dca' and abs(r['bitrate'] - br) < 0.15*br:
                if 'embed-fro-err' in r.keys():
                    frodist = math.sqrt(r['embed-fro-err'])
                    br_2_mks[br].append((r['m'],r['k'],frodist))
                    br_2_mks[br].sort(key=lambda x:x[1])
    return br_2_mks

def get_dca_best_params(results, bitrates, base):
    '''A simple query routine for getting the best parameters in a dca tune experiment'''
    res = results
    br_2_params = dict()
    for br in bitrates:
        lowest_loss = float('inf')
        best_res = None
        for r in res:
            if r == {} or r['base'] != base: continue
            if r['method'] == 'dca' and  r['ibr'] == br:
                if lowest_loss > r['embed-frob-err']:
                    lowest_loss = r['embed-frob-err']
                    best_res = r
        br_2_params[br] = (np.sqrt(lowest_loss),
                            best_res['m'], 
                            best_res['k'], 
                            best_res['lr'], 
                            best_res['batchsize'], 
                            best_res['tau'],
                            best_res['gradclip'])
    return br_2_params

def histogram(embpath,name):
    '''plots histogram of npy embeddings'''
    X,v = load_embeddings(embpath)
    plt.hist(X,bins=100)
    plt.savefig(str(pathlib.PurePath(get_plots_path(),
        f"{get_date_str()}:embeddings-histogram:{name}")))
    plt.close()
    

def prep_sentiment_results(results):
    '''
    Since sentiment data requires average over data sets and plotting for three differen models,
    I use this prep routine to convert the data into a format that works with rest of plotting pipeline
    Input: results -- list of json results
    Output: prepped_results -- same list of json results, with additional entries used for plotting 
    NO modification to original values
    '''
    for res in results:
        scores_for_sentiment = dict()
        for key in res.keys():
            if 'sentiment-score' in key:
                _,_,model,dataset = key.split('-')
                if model in scores_for_sentiment.keys():
                    scores_for_sentiment[model].extend(scores_for_sentiment[model])
                else:
                     scores_for_sentiment[model] = [res[key]]
        for score_key in scores_for_sentiment.keys():
            res['avg-sentiment-%s' % score_key] = \
            1 - sum(scores_for_sentiment[score_key])/len(scores_for_sentiment[score_key])
    return results

def prep_codebook_free_bitrate_results(results):
    for res in results:
        if res['method'] == 'dca':
            res['bitrate-codes-only'] = res['vocab']*res['m']*np.log2(res['k'])/(res['vocab']*res['dim'])
        elif res['method'] == 'kmeans':
            res['bitrate-codes-only'] = res['bitsperblock']/res['blocklen']
    return results

def prep_dca_br_correction_results(results):
    for res in results:
        print(res)
        if res['method'] == 'dca':
            res['bitrate'] = (res['m'] * res['vocab'] * np.log2(res['k']) + 32 * res['m']*res['k']*res['dim'])/(res['vocab']*res['dim'])
    return results

def make_plots( x,
                y,
                results,
                source,
                vocab,
                methods=['dca', 'kmeans'], 
                include_baseline=False, 
                xscale='linear',
                yscale='linear',
                xticks=[0.1,0.25,0.5,1,2,4],
                lbl_size=12):
    plt.figure(figsize=(14,8))
    plt.tick_params(axis='both', which='major', labelsize=lbl_size)

    for method in methods:
        if method in special_treatment_methods(): # some methods need special treatment
            if method == 'tuned-dca':
                br_2_params = get_dca_best_params(agg('merged-experiment5-dca-hp-tune/*'), [0.1,0.25,0.5,1,2,4], source)
                data_x = sorted(list(br_2_params.keys()))
                data_y = [br_2_params[x][0] for x in data_x]
                plt.plot(data_x, data_y, color=color_lookup(method), linewidth=3.0, label=method)
            else:
                raise ValueError('method identified as special treatment, but not supported in code')
            continue
        data = get_all_data(results, source, vocab, method, x, y)
        data_x,data_y = compute_avg_variable_len(data)
        errbars_low_abs, errbars_high_abs = compute_min_max_variable_len(data)
        errbars_low_rel = np.array(data_y) - np.array(errbars_low_abs)
        errbars_high_rel =  np.array(errbars_high_abs) - np.array(data_y)
        errbars = np.array([errbars_low_rel, errbars_high_rel])
        plt.errorbar(data_x, data_y, fmt=color_lookup(method), yerr=errbars, linewidth=3.0, label=method)
    if include_baseline: #hardcoded for now -- needs a fix
        print(results)
        data = get_all_data(results, source, vocab, 'baseline', x, y)
        vals = data[32.0]
        data_x = xticks
        data_y = [np.mean(np.array(vals))]*len(xticks)
        errbar = 0.5*(max(vals) - min(vals)) #TODO fix this weird error bar centering
        plt.errorbar(data_x, data_y, fmt=color_lookup('baseline'), yerr=errbar, label='baseline (32-bit)', linewidth=3.0, linestyle='--')
    plt.xlabel(nice_names_lookup(x), size=lbl_size)
    plt.ylabel(nice_names_lookup(y), size=lbl_size)
    plt.xscale(xscale)
    plt.yscale(yscale)
    xticks_lbls = [str(i) for i in xticks]
    plt.xticks(xticks,[str(i) for i in xticks])
    plt.title('%s vs. %s for %s' % (nice_names_lookup(x),
                                    nice_names_lookup(y),
                                    nice_names_lookup(source)))
    plt.legend(fontsize='x-large')
    #plt.figure(figsize=(8,8))
    #plt.tight_layout()
    plt.savefig(str(pathlib.PurePath(get_plots_path(),
        '%s:%s-vs-%s_%s,%s,%s-%s' % (get_date_str(),x, y, source, vocab, xscale, yscale))))
    plt.close()

def color_lookup(method):
    colors = dict()
    colors['dca'] = 'r'
    colors['kmeans'] = 'b'
    colors['baseline'] = 'c'
    colors['stochround'] = 'm'
    colors['tuned-dca'] = 'm'
    colors['clipnoquant'] = 'y'
    colors['midriser'] = 'g'
    colors['optranuni'] = 'g'
    colors['dim-reduc'] = 'm'
    assert method in colors.keys(), "A color has not been designated for the requested method"
    return colors[method]

def xy_dataset_qry_lookup(x,y,method=None):
    qry = 'merged-experiment2-5X-seeds/*', 198
    if 'maketime' in y: 
        qry = 'merged-experiment4-1X-seeds/*', 90
    return qry

def xy_ticks_lookup(x,y):
    xticks = None
    yticks = None
    if y == 'embed-fro-dist':
        xticks = [0.1,0.25,0.5,1,2,4,8]
        yticks = [100,500,2500,5000]
    return xticks, yticks

def special_treatment_methods():
    return ['tuned-dca']

def nice_names_lookup(ugly_name):
    ugly_2_nice = dict()
    ugly_2_nice['max-f1'] = 'QA F1 Score'
    ugly_2_nice['semantic-dist'] = 'Average Cosine Distance'
    ugly_2_nice['analogy-avg-score'] = 'Aggregate Word Analogy Accu.'
    ugly_2_nice['similarity-avg-score'] = 'Aggregate Word Sim. Spearman Corr.'
    ugly_2_nice['embed-fro-dist'] = 'Frobenius Distance'
    ugly_2_nice['bitrate'] = 'Bitrate'
    ugly_2_nice['embed-maketime-secs'] = 'Compression Runtime (secs)'
    ugly_2_nice['glove'] = 'GloVe'
    ugly_2_nice['fasttext'] = 'FastText'
    ugly_2_nice['avg-sentiment-lstm'] = 'Agg. Sentiment Analysis Accu. with LSTM'
    ugly_2_nice['avg-sentiment-cnn'] = 'Agg. Sentiment Analysis Accu. with CNN'
    ugly_2_nice['avg-sentiment-la'] = 'Agg. Sentiment Analysis Accu. with Perceptron'

    if ugly_name in ugly_2_nice.keys():
        return ugly_2_nice[ugly_name]
    else:
        return ugly_name
