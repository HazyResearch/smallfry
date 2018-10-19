import sys
import os
import argh
import pathlib
import numpy as np
from plot_tools import *
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) #FOR LOCAL IMPORTS
from experimental_utils import *

def results_aggregator(rungroup,expected_num_res):
    results = agg(f"{rungroup}/*",expected_num_res=expected_num_res)
    respath = str(pathlib.PurePath(get_agg_results_path(),rungroup))
    save_dict_as_json(results, respath)

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
    methods = ['dca','kmeans','stochround','midriser']
    core_plotter(x,y,sources,vocabs,methods,lambda x: x)

def plot_midriser():
    x = ['bitrate','bitrate','bitrate','bitrate','bitrate','bitrate']
    y = ['embed-fro-dist','similarity-avg-score','analogy-avg-score','max-f1','semantic-dist','embed-maketime-secs']
    sources = ['glove', 'fasttext']
    vocabs = [400000]
    methods = ['midriser']
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
    
def list_best_dca():
    results = agg('merged-experiment5-dca-hp-tune/*',expected_num_res=1296)
    br_2_params = get_dca_best_params(results,[0.1,0.25,0.5,1,2,4], 'glove')
    print(br_2_params)

def plot_frobenius():
    x = ['bitrate']
    y = ['embed-fro-dist']
    sources = ['glove', 'fasttext']
    vocabs = [400000]
    methods = ['dca','kmeans','tuned-dca','optranuni']
    core_plotter(x,y,sources,vocabs,methods, prep_dca_br_correction_results)

def plot_exp6():
    results = agg('2018-10-09-experiment6-dim-reduc-mini/*',expected_num_res=19)
    print(results)
    for result in results:
        if result['method'] == 'glove':
            result['bitrate'] = result['dim']/300*32
            result['base'] = 'glove'
            result['vocab'] = 71291
            if result['dim'] < 300:
                result['method'] = 'dim-reduc'
            else:
                result['method'] = 'baseline'
                print(result)
        
    vocabs = [71291]
    x = ['bitrate','bitrate']
    y = ['similarity-avg-score', 'analogy-avg-score']
    sources = ['glove']
    methods = ['dim-reduc', 'kmeans', 'optranuni']
    for i in range(len(x)):
        for source in sources:
            for vocab in vocabs:
                for scales in [ ('linear','linear'),('log','linear'),('linear','log'),('log','log') ]:
                    make_plots(x[i],y[i],results,source,vocab,methods=methods,
                        include_baseline=True,xscale=scales[0],yscale=scales[1])    


def plot_exp7():
    results = agg('2018-10-11-experiment7-quant-ablation/*',expected_num_res=19)
    newresults = []
    for result in results:
        if result['method'] == 'glove':
            result['bitrate'] = result['dim']/300*32
            result['base'] = 'glove'
            result['vocab'] = 71291
            if result['dim'] < 300:
                result['method'] = 'dim-reduc'
            else:
                result['method'] = 'baseline'
        elif result['method'] == 'clipnoquant':
            result['bitrate'] = result['ibr']
            newresults.append(result)
        else:
            newresults.append(result)
        
    vocabs = [71291]
    x = ['bitrate']
    y = ['embed-frob-dist']
    sources = ['glove']
    methods = ['kmeans', 'optranuni','clipnoquant']
    for i in range(len(x)):
        for source in sources:
            for vocab in vocabs:
                for scales in [ ('linear','linear'),('log','linear'),('linear','log'),('log','log') ]:
                    results = prep_sentiment_results(results)
                    make_plots(x[i],y[i],results,source,vocab,methods=methods,
                        include_baseline=True,xscale=scales[0],yscale=scales[1],xticks=[0.5,1,2,4,8]) 

def plot_exp9():
    results = agg('2018-10-16-exp9-dim-vs-prec-quantized/*',expected_num_res=15)
    def prep_exp9_results(results):
        for result in results:
            if result['memory'] > 22813153:
                result['method'] = 'high-mem'
            elif result['memory'] < 22813151:
                result['method'] = 'low-mem'
            else:
                result['method'] = 'mid-mem'
        return results

    vocab = 71291
    x = ['bitrate','bitrate']
    y = ['similarity-avg-score', 'analogy-avg-score']
    source = 'glove'
    methods = ['high-mem','mid-mem','low-mem']
    for i in range(len(x)):
        for method in methods:
            for scales in [ ('linear','linear'),('log','linear'),('linear','log'),('log','log') ]:
                    make_plots(x[i],y[i],prep_exp9_results(results),source,vocab,methods=methods,
                        include_baseline=False,xscale=scales[0],yscale=scales[1],xticks=[1,2,4,8,32]) 

def plot_exp11():
    results = agg('2018-10-17-exp11-stoch-benchmarks/*',expected_num_res=13)
    x = ['ibr','ibr','ibr']
    y = ['similarity-avg-score','analogy-avg-score','embed-frob-dist']
    source = 'glove'
    vocab = 400000
    methods = ['clipnoquant','stochoptranuni','optranuni','kmeans']
    for i in range(len(x)):
        for method in methods:
            for scales in [ ('linear','linear'),('log','linear'),('linear','log'),('log','log') ]:
                    make_plots(x[i],y[i],results,source,vocab,methods=methods,
                        include_baseline=True,xscale=scales[0],yscale=scales[1],xticks=[1,2,4]) 

def exp5_dca_hp_results_aggregator():
    results_aggregator('merged-experiment5-dca-hp-tune/*',expected_num_res=1296)

def plot_exp5_lr():
    rg = 'merged-experiment5-dca-hp-tune'
    results = import_results(rg)
    print(agg(rg))
    results.extend(agg(f"{rg}/*"))
    #define defaults
    defaults = dict()
    defaults['tau'] = 1
    defaults['gradclip'] = 0.001
    defaults['m'] = 573
    defaults['k'] = 4
    defaults['batchsize'] = 64
    def prep_dca_lr_sweep_results(results):
        matchup_results = []
        for result in results:
            matchup = True
            for default in defaults.keys():
                if not result: 
                    matchup = False
                    break
                if defaults[default] != result[default]:
                    matchup = False
            if matchup:
                result['Frobenius-Distance'] = np.sqrt(result['embed-frob-err'])
                matchup_results.append(result)
        return matchup_results
    
    x = 'lr'
    y = 'Frobenius-Distance'
    source = 'glove'
    methods = ['dca']
    vocab = 400000
    for scales in [ ('log','linear'),('log','log') ]:
                    make_plots(x,y,prep_dca_lr_sweep_results(results),source,vocab,methods=methods,
                        include_baseline=False,xscale=scales[0],yscale=scales[1],xticks=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1])

def plot_exp5_tune_metrics():
    rg = 'merged-experiment5-dca-hp-tune'
    results = import_results(rg)
    results.extend(agg(f"{rg}/*"))
    def prep_dca_stats_results(results):
        scores = []
        for result in results:
            if 'ibr' in result.keys() and result['ibr'] == 4 and result['base'] == 'glove':
                if np.sqrt(result['embed-frob-err']) < 1100:
                    print(result)
                scores.append(np.sqrt(result['embed-frob-err']))
        return scores
    scores = prep_dca_stats_results(results)
    print(len(scores))
    print(np.mean(scores))
    print(np.var(scores))
    print(np.max(scores))
    print(np.min(scores))

def plot_exp8():
    results = agg('merged-exp8-wiki-trained/*',expected_num_res=13)
    print(results)
    def prep_exp8_results(results):
        for result in results:
            if result['method'] == 'glove':
                result['bitrate'] = result['dim']/320*32
                result['base'] = 'glove'
                result['method'] = 'dim-reduc' if result['bitrate'] < 30 else 'baseline'
        return results

    x = ['bitrate','bitrate','bitrate']
    y = ['similarity-avg-score','analogy-avg-score','max-f1']
    source = 'glove'
    vocab = 3801686
    methods = ['dim-reduc','optranuni','kmeans','naiveuni']
    for i in range(len(x)):
        for method in methods:
            for scales in [ ('linear','linear'),('log','linear'),('linear','log'),('log','log') ]:
                    make_plots(x[i],y[i],prep_exp8_results(results),source,vocab,methods=methods,
                        include_baseline=True,xscale=scales[0],yscale=scales[1],xticks=[1,2,4]) 

def viz_exp8():
    results = agg('merged-exp8-wiki-trained/*',expected_num_res=10)
    print(results)
    def prep_exp8_results(results):
        for result in results:
            if result['method'] == 'glove':
                result['bitrate'] = result['dim']/320*32
                result['base'] = 'glove'
                result['method'] = 'dim-reduc' if result['bitrate'] < 30 else 'baseline'
        return results

    x = ['bitrate','bitrate','bitrate']
    y = ['similarity-avg-score','analogy-avg-score','max-f1']
    source = 'glove'
    vocab = 3801686
    methods = ['dim-reduc','optranuni','kmeans']
    for i in range(len(x)):
        for method in methods:
            for scales in [ ('linear','linear'),('log','linear'),('linear','log'),('log','log') ]:
                    make_plots(x[i],y[i],prep_exp8_results(results),source,vocab,methods=methods,
                        include_baseline=True,xscale=scales[0],yscale=scales[1],xticks=[1,2,4]) 
        


parser = argh.ArghParser()
parser.add_commands([plot_embeddings_battery, 
                    plot_embeddings_sentiment,
                    plot_embeddings_bitrate_codes_only,
                    plot_midriser,
                    plot_histograms,
                    list_best_dca,
                    plot_frobenius,
                    plot_exp6,
                    plot_exp7,
                    plot_exp11,
                    plot_exp8,
                    plot_exp5_lr,
                    exp5_dca_hp_results_aggregator,
                    plot_exp5_tune_metrics,
                    plot_exp9])

if __name__ == '__main__':
    parser.dispatch()
