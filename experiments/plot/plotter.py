import sys
import os
import argh
import matplotlib.pyplot as plt
from plot_tools import *
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) #FOR LOCAL IMPORTS
from experimental_utils import *

def get_dca_params(results, bitrates, base):
    res = results
    br_2_mks = dict()
    for br in bitrates:
        br_2_mks[br] = []
        for r in res:
            if r == {} or r['base'] != base: continue
            if r['method'] == 'dca' and abs(r['bitrate'] - br) < 0.15*br:
                if 'embed-fro-dist' in r.keys():
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

def plot_embeddings_frobenius(qry='merged-experiment2-5X-seeds/*',seeds=[4974,6117],lbl_size=17):
    x = ['bitrate','bitrate','bitrate','bitrate']
    y = ['embed-fro-dist','similarity-avg-score','analogy-avg-score','max-f1','semantic-dist']
    sources = ['glove', 'fasttext']
    vocabs = [400000]
    methods = ['dca','kmeans']
    for i in range(len(x)):
        for source in sources:
            for vocab in vocabs:
                for method in methods:
                    results = agg(qry,130)
                    data = get_data(results, source, vocab, method, seeds, x, y)
                    plt.tick_params(axis='both', which='major', labelsize=lbl_size)
                    plt.errorbar(qa_dca_x, qa_dca_y, fmt='r',yerr=dca_err, linewidth=3.0, label='deep autoencoder')
                plt.errorbar(qa_lmq_x, qa_lmq_y, fmt='b',yerr=lmq_err, linewidth=3.0, label='k-means'





parser = argh.ArghParser()
parser.add_commands([eval_embeddings])

if __name__ == '__main__':
    parser.dispatch()
