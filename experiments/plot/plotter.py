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


parser = argh.ArghParser()
parser.add_commands([plot_embeddings_battery, 
                    plot_embeddings_sentiment,
                    plot_embeddings_bitrate_codes_only,
                    plot_midriser,
                    plot_histograms])

if __name__ == '__main__':
    parser.dispatch()
