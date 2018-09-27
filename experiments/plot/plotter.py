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

def plot_embeddings_frobenius(qry='merged-experiment2-5X-seeds/*',seeds=[4974,6117],lbl_size=12):
    x = ['bitrate','bitrate','bitrate','bitrate','bitrate','bitrate']
    y = ['embed-fro-dist','similarity-avg-score','analogy-avg-score','max-f1','semantic-dist','maketime-secs']
    sources = ['glove', 'fasttext']
    vocabs = [400000]
    methods = ['dca','kmeans']
    for i in range(len(x)):
        for source in sources:
            for vocab in vocabs:
                data = dict()
                plt.tick_params(axis='both', which='major', labelsize=lbl_size)
                for method in methods:
                    data = get_all_data(agg(qry,expected_num_res=130), source, vocab, method, x[i], y[i])
                    data_x,data_y = compute_avg_variable_len(data)
                    _,errbars = compute_avg_variable_len(data)
                    color = 'r' if method == 'dca' else 'b'
                    plt.errorbar(data_x, data_y, fmt=color, yerr=errbars, linewidth=3.0, label=method)
                #plt.axhline(y=np.mean(baselines),linestyle='--',label='baseline (32-bit)',linewidth=3.0)
                plt.xlabel(nice_names_lookup(x[i]), size=lbl_size)
                plt.ylabel(nice_names_lookup(y[i]), size=lbl_size)
                plt.title('%s vs. %s' % (nice_names_lookup(x[i]),nice_names_lookup(y[i]))
                plt.legend(fontsize='x-large')
                plt.tight_layout()
                plt.savefig(str(pathlib.PurePath(get_plots_path(),'%s-vs-%s_%s,%s' % (x[i], y[i], source, vocab))))
                plt.close()

parser = argh.ArghParser()
parser.add_commands([plot_embeddings_frobenius])

if __name__ == '__main__':
    parser.dispatch()
