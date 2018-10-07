import sys
import os
import argh
import numpy as np
from plot_tools import *
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) #FOR LOCAL IMPORTS
from experimental_utils import *

def core_plotter(x,y,sources,vocabs,methods,results_prep):
    for i in range(len(x)):
        for source in sources:
            for vocab in vocabs:
                needs_baseline = y[i] in ['similarity-avg-score','analogy-avg-score','max-f1','avg-sentiment-lstm','avg-sentiment-la','avg-sentiment-cnn']
                qry, expected_num = xy_dataset_qry_lookup(x[i],y[i])
                results = agg(qry,expected_num_res=expected_num)
                results = results_prep(results)
                for scales in [ ('linear','linear'),('log','linear'),('linear','log'),('log','log') ]:
                    make_plots(x[i],y[i],results,source,vocab,methods=methods,
                        include_baseline=needs_baseline,xscale=scales[0],yscale=scales[1])

def plot_embeddings_battery():
    x = ['bitrate','bitrate','bitrate','bitrate','bitrate','bitrate']
    y = ['embed-fro-dist','similarity-avg-score','analogy-avg-score','max-f1','semantic-dist','embed-maketime-secs']
    sources = ['glove', 'fasttext']
    vocabs = [400000]
    methods = ['dca','kmeans','stochround']
    core_plotter(x,y,sources,vocabs,methods,lambda x: x)
    
def plot_embeddings_sentiment():
    x = ['bitrate','bitrate','bitrate']
    y = ['avg-sentiment-lstm','avg-sentiment-cnn','avg-sentiment-la']
    sources = ['glove', 'fasttext']
    vocabs = [400000]
    methods = ['dca','kmeans']
    core_plotter(x,y,sources,vocabs,methods,prep_sentiment_results)

def plot_embeddings_bitrate_codes_only():
    x = ['bitrate-codes-only']
    y = ['embed-fro-dist']
    sources = ['glove', 'fasttext']
    vocabs = [400000]
    methods = ['dca','kmeans']
    core_plotter(x,y,sources,vocabs,methods,prep_codebook_free_bitrate_results)

def plot_histograms():
    ft_path = str(pathlib.PurePath(get_base_embed_path_head(),'fasttext_k=400000'))
    histogram(ft_path,'fasttext')
    glove_path = str(pathlib.PurePath(get_base_embed_path_head(),'glove_k=400000'))
    histogram(glove_path, 'glove')

'''
def plot_embeddings_battery_old(qry='merged-experiment2-5X-seeds/*',seeds=[4974,6117],lbl_size=12):
    x = ['bitrate','bitrate','bitrate','bitrate','bitrate','bitrate']
    y = ['embed-fro-dist','similarity-avg-score','analogy-avg-score','max-f1','semantic-dist','maketime-secs']
    #x = ['bitrate']
    #y = ['semantic-dist']
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
                    #if y[i] == 'maketimes-secs':
                    #    print(data)
                    data_x,data_y = compute_avg_variable_len(data)
                    errbars_low_abs, errbars_high_abs = compute_std_variable_len(data)
                    errbars_low_rel = np.array(data_y) - np.array(errbars_low_abs)
                    errbars_high_rel = np.array(errbars_high_abs) - np.array(data_y)
                    errbars = np.array([errbars_low_rel, errbars_high_rel])
                    #print(errbars_low)
                    #print(errbars_high)
                    color = 'r' if method == 'dca' else 'b'
                    plt.errorbar(data_x, data_y, fmt=color, yerr=errbars, linewidth=3.0, label=method)
                #plt.axhline(y=np.mean(baselines),linestyle='--',label='baseline (32-bit)',linewidth=3.0)i
                if y[i] in ['max-f1', 'analogy-avg-score', 'similarity-avg-score']:
                    data = get_all_data(agg(qry,expected_num_res=130), source, vocab, 'baseline', x[i], y[i])
                    print(data)
                    vals = data[32.0]

                    data_x = [0.1,0.25,0.5,1,2,4]
                    data_y = [np.mean(np.array(vals))]*6
                    errbar = 0.5*(max(vals) - min(vals))
                    print(data_x)
                    print(data_y)
                    plt.errorbar(data_x, data_y, fmt='c', yerr=errbar, label='baseline (32-bit)', linewidth=3.0, linestyle='--')
                #plt.axhline(y=data_y,linestyle='--',label='baseline (32-bit)',linewidth=3.0)
                plt.xlabel(nice_names_lookup(x[i]), size=lbl_size)
                plt.ylabel(nice_names_lookup(y[i]), size=lbl_size)
                plt.yscale('log')
                plt.xticks([0.1,0.25,0.5,1,2,4],['0.1','0.25','0.5','1','2','4'])
                plt.title('%s vs. %s for %s' % (nice_names_lookup(x[i]),
                                                    nice_names_lookup(y[i]),
                                                    nice_names_lookup(source)))
                plt.legend(fontsize='x-large')
                plt.tight_layout()
                plt.savefig(str(pathlib.PurePath(get_plots_path(),'%s-vs-%s_%s,%s' % (x[i], y[i], source, vocab))))
                plt.close()
'''
parser = argh.ArghParser()
parser.add_commands([plot_embeddings_battery, plot_embeddings_sentiment,plot_embeddings_bitrate_codes_only])

if __name__ == '__main__':
    parser.dispatch()
