import glob
import pathlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import utils

default_var_info = ['gitdiff',['']]

# Returns a list of result dictionaries whose filenames match the path_regex.
def gather_results(path_regex):
    file_list = glob.glob(path_regex)
    return [utils.load_from_json(f) for f in file_list]

def clean_results(results):
    cleaned = []
    for result in results:
        result = flatten_dict(result)
        if result['compresstype'] == 'nocompress' and result['embedtype'] == 'glove400k':
            # NOTE: This assumes all other compression methods are compressing
            # a 300 dimensional embedding
            effective_bitrate = (32.0/300.0) * result['embeddim']
            result['compression-ratio'] = 32.0/effective_bitrate
        else:
            result['compression-ratio'] = 32.0/result['bitrate']
        if result['compresstype'] in ['uniform','nocompress','kmeans']:
            vocab = utils.get_embedding_vocab(result['embedtype'])
            result['memory'] = vocab * result['embeddim'] * result['bitrate']
        if 'test-err' in result:
            result['test-acc'] = 1-result['test-err']
            result['val-acc'] = 1-result['val-err']
        cleaned.append(result)
    return cleaned

# Note: this only flattens one layer down
def flatten_dict(to_flatten):
    flattened = {}
    for k,v in to_flatten.items():
        if isinstance(v,dict):
            for k2,v2 in v.items():
                flattened[k2] = v2
        else:
            flattened[k] = v
    return flattened

# Returns a list of result dictionaries with the subset of results from
# all_results which exactly matched the 'key_values_to_match' dictionary.
def extract_result_subset(all_results, key_values_to_match):
    subset = []
    for result in all_results:
        if matches_all_key_values(result, key_values_to_match):
            subset.append(result)
    return subset

# return True if result[key] in values for all key-value pairs in key_values_to_match
def matches_all_key_values(result, key_values_to_match):
    for key,values in key_values_to_match.items():
        if result[key] not in values: return False
    return True

# TODO: add error bar support
def plot_driver(all_results, key_values_to_match, info_per_line, x_metric, y_metric,
                logx=False, logy=False, title=None, var_info=default_var_info,
                csv_file=None):
    if len(key_values_to_match) == 0:
        subset = all_results
    else:
        subset = extract_result_subset(all_results, key_values_to_match)
    lines = extract_x_y_foreach_line(subset, info_per_line, x_metric, y_metric, var_info=var_info)
    plot_lines(lines, x_metric, y_metric, logx=logx, logy=logy, title=title, csv_file=csv_file)

# lines is a dictionary of {line_name:(x,y)} pairs, where x and y are numpy
# arrays with the x and y values to be plotted.
def plot_lines(lines, x_metric, y_metric, logx=False, logy=False, title=None, csv_file=None):
    f = None
    if csv_file:
        f = open(csv_file,'w+')
    legend = []
    for line_name,xy in lines.items():
        legend.append(line_name)
        # plt.plot(xy[0],xy[1],'o--')
        sorted_x = xy[0]
        y_array = xy[1]
        y_avg = np.average(y_array,axis=0)
        y_std = np.std(y_array,axis=0)
        # if line_name == 'Dim. reduction':
        #     plt.errorbar(sorted_x, y_avg, yerr=y_std, marker='o', capthick=4, capsize=10000)
        # else:
        plt.errorbar(sorted_x, y_avg, yerr=y_std, marker='o', capthick=4, capsize=10)
        if f:
            f.write('{}\n'.format(line_name))
            f.write(x_metric + ',' + ','.join([str(a) for a in sorted_x.tolist()]) + '\n')
            for i in range(y_array.shape[0]):
                y = y_array[i,:]
                f.write('{} ({}),{}\n'.format(y_metric,i, ','.join([str(a) for a in y.tolist()])))
            f.write(y_metric + ' (avg.),' + ','.join([str(a) for a in y_avg.tolist()]) + '\n')
            f.write(y_metric + ' (st. dev.),' + ','.join([str(a) for a in y_std.tolist()]) + '\n')
    if f:
        f.close()

    plt.legend(legend)
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    if logx: plt.xscale('log')
    if logy: plt.yscale('log')
    if title:
        plt.title(title)
    else:
        plt.title('{} vs {}'.format(y_metric, x_metric))

# from a list of results, extracts the x,y info for each line.
# Specifically, for each line, extracts the subset of results corresponding
# to that line (based on key-value pairs matching the dict specified in
# info_per_line), and then extracts x,y arrays from these results.
def extract_x_y_foreach_line(results, info_per_line, x_metric, y_metric, var_info=default_var_info):
    lines = {}
    for line_name,key_values in info_per_line.items():
        line_subset = extract_result_subset(results, key_values)
        lines[line_name] = get_x_y_values(line_subset, x_metric, y_metric, var_info=var_info)
    return lines

# extracts x,y arrays for a specific line_subset
def get_x_y_values(line_subset, x_metric, y_metric, var_info=default_var_info):
    var_key = var_info[0]
    var_values = var_info[1]
    x = {}
    y = {}
    for val in var_values:
        x[val] = []
        y[val] = []
    for result in line_subset:
        val = result[var_key]
        if val in var_values:
            x[val].append(result[x_metric])
            y[val].append(result[y_metric])
    for val in var_values:
        x[val] = np.array(x[val])
        y[val] = np.array(y[val])
        ind = np.argsort(x[val])
        x[val] = x[val][ind]
        y[val] = y[val][ind]
    sorted_x = x[var_values[0]]
    for val in var_values:
        assert np.array_equal(sorted_x, x[val])
    y_array = np.zeros((len(var_values), len(sorted_x)))
    for i,val in enumerate(var_values):
        y_array[i,:] = y[val]
    return sorted_x, y_array

def plot_frob_squared_vs_bitrate():
    path_regex = str(pathlib.PurePath(utils.get_base_dir(), 'embeddings',
                     'glove400k', 'round1_tuneDCA_results', '*final.json'))
    all_results = gather_results(path_regex)
    plt.figure(1)
    plot_driver(all_results, {'compresstype':['kmeans','uniform','dca']},
        {'kmeans':
            {
                'compresstype':['kmeans']
            },
        'uniform (adaptive-stoch)':
            {
                'compresstype':['uniform'],
                'adaptive':[True],
                'stoch':[True],
                'skipquant':[False]
            },
        'uniform (adaptive-det)':
            {
                'compresstype':['uniform'],
                'adaptive':[True],
                'stoch':[False],
                'skipquant':[False]
            },
        'uniform (adaptive-skipquant)':
            {
                'compresstype':['uniform'],
                'adaptive':[True],
                'stoch':[False],
                'skipquant':[True]
            },
        # 'uniform (non-adaptive, det)':
        #     {
        #         'compresstype':['uniform'],
        #         'adaptive':[False],
        #         'stoch':[False],
        #         'skipquant':[False]
        #     },
        'dca (k=4,lr=0.0003)':
            {
                'compresstype':['dca'],
                'k':[4],
                'lr':[0.0003]
            }
        },
        'bitrate',
        'frob-squared-error',
        logy=True
    )
    plt.show()

def plot_dca_frob_squared_vs_lr(results_path):
    # path_regex = str(pathlib.PurePath(utils.get_base_dir(), 'embeddings',
    #                  'glove400k', 'round1_tuneDCA_results', '*final.json'))
    # all_results = gather_results(path_regex)
    all_results = utils.load_from_json(results_path)
    embedtype = all_results[0]['embedtype']
    bitrates = [1,2,4] # 3
    ks = [2,4,8,16] # 4
    # lrs = ['0.00001', '0.00003', '0.0001', '0.0003', '0.001'] # 5
    plt.figure(1)
    for i,b in enumerate(bitrates):
        info_per_line = {}
        for k in ks:
            info_per_line[str(k)] = {'k':[k]}
        plt.subplot(311 + i)
        plot_driver(all_results, {'compresstype':['dca'],'bitrate':[b]},
            info_per_line,
            'lr',
            'frob-squared-error',
            logx=True,
            logy=True,
            title='{}, bitrate = {}, lr vs. frob'.format(embedtype, b)
        )
    plt.show()

def dca_get_best_k_lr_per_bitrate(path_regex):
    # path_regex1 = str(pathlib.PurePath(utils.get_base_dir(), 'embeddings',
    #                 'glove400k', 'round1_tuneDCA_results', '*final.json'))
    # path_regex2 = '/proj/smallfry/embeddings/fasttext1m/2018-12-16-fasttextTuneDCA/*/*final.json'
    # best = plotter.dca_get_best_k_lr_per_bitrate(path_regex)
    all_results = clean_results(gather_results(path_regex))
    bitrates = [1,2,4] # 3
    # ks = [2,4,8,16] # 4
    # lrs = ['0.00001', '0.00003', '0.0001', '0.0003', '0.001'] # 5
    best_k_lr_per_bitrate = {}
    for b in bitrates:
        dca_bitrate_results = extract_result_subset(all_results, {'compresstype':['dca'],'bitrate':[b]})
        best = np.inf
        for result in dca_bitrate_results:
            if result['frob-squared-error'] < best:
                best = result['frob-squared-error']
                best_k_lr_per_bitrate[b] = {'k':result['k'], 'lr':result['lr'], }
    return best_k_lr_per_bitrate

def plot_embedding_spectra():
    path = str(pathlib.PurePath(utils.get_base_dir(), 'base_embeddings',
               'glove400k', 'glove.6B.{}d.txt'))
    output_file_str = str(pathlib.PurePath(utils.get_git_dir(),
        'paper', 'figures', '{}_spectra.pdf'))

    ds = [50,100,200,300]
    for d in ds:
        emb,_ = utils.load_embeddings(path.format(d))
        s = np.linalg.svd(emb,compute_uv=False,full_matrices=False)    
        plt.plot(s)
    plt.title('Glove400k spectra')
    plt.yscale('log')
    plt.ylabel('Singular values')
    plt.legend(['d=' + str(d) for d in ds])
    plt.savefig(output_file_str.format('glove400k'))

    plt.figure(2)
    path = str(pathlib.PurePath(utils.get_base_dir(), 'base_embeddings',
               'fasttext1m', 'wiki-news-300d-1M.vec'))
    emb,_ = utils.load_embeddings(path)
    s = np.linalg.svd(emb,compute_uv=False,full_matrices=False)
    plt.title('fasttext1m, d=300')
    plt.plot(s)
    plt.yscale('log')
    plt.ylabel('Singular values')
    plt.savefig(output_file_str.format('fasttext1m'))

def gather_ICML_results():
    embedtypes = ['glove-wiki400k-am','glove400k','fasttext1m']
    result_file_regexes = ['*evaltype,qa*final.json', '*evaltype,sent*lr,0*final.json',
            '*evaltype,intrinsics*final.json', '*evaltype,synthetics*final.json']
    # if we want the compression config file, use 'embedtype,*final.json'
    path_regex = '/proj/smallfry/embeddings/{}/*/*/{}'
    all_results = []
    for embedtype in embedtypes:
        for result_file_regex in result_file_regexes:
            results = gather_results(path_regex.format(embedtype, result_file_regex))
            print('{}, {}, {}'.format(embedtype, result_file_regex, len(results)))
            all_results.extend(results)
    result_dir = '/proj/smallfry/results/'
    utils.save_to_json(all_results, result_dir + 'ICML_results.json')

def get_best_lr_sentiment():
    path_regex = '/proj/smallfry/embeddings/*/*/*/*evaltype,sent*final.json'
    all_results = clean_results(gather_results(path_regex))
    # first gather list of base_embeds
    base_embeds = []
    for result in all_results:
        if result['base-embed-path'] not in base_embeds:
            base_embeds.append(result['base-embed-path'])
    assert len(base_embeds) == 11
    # now find best lr per base_embed, based on average of validation errors.
    datasets = ['mr','subj','cr','sst','trec','mpqa']
    lrs = all_results[0]['lrs']
    assert len(lrs) == 7
    num_seeds = 5
    best_lr_array = np.zeros((len(base_embeds),len(datasets)))
    best_lr_dict = {}
    results_array = np.zeros((len(base_embeds),len(datasets),len(lrs)))
    val_errs = np.zeros((num_seeds,len(lrs)))
    for i,base_embed in enumerate(base_embeds):
        best_lr_dict[base_embed] = {}
        for j,dataset in enumerate(datasets):
            base_embed_results = extract_result_subset(all_results,
                {'base-embed-path':[base_embed], 'dataset':[dataset]})
            assert len(base_embed_results) == num_seeds
            for k in range(num_seeds):
                val_errs[k,:] = base_embed_results[k]['val-errs']
            avg_val_errs = np.mean(val_errs,axis=0)
            ind = np.argmin(avg_val_errs)
            results_array[i,j,:] = avg_val_errs
            best_lr_array[i,j] = lrs[ind]
            best_lr_dict[base_embed][dataset] = lrs[ind]
    lr_tuning_results = {
        'best_lr_dict': best_lr_dict,
        'best_lr_array': best_lr_array.tolist(),
        'results_array': results_array.tolist(),
        'base_embeds': base_embeds,
        'datasets': datasets,
        'lrs': lrs
    }
    return lr_tuning_results


def plot_ICML_results(embedtype, evaltype, y_metric, dataset=None):
    # load and clean all results
    results_file = str(pathlib.PurePath(utils.get_base_dir(), 'results', 'ICML_results.json'))
    all_results = utils.load_from_json(results_file)
    all_results = clean_results(all_results)

    # prepare filenames of output csv and pdf files.
    output_file_str = str(pathlib.PurePath(utils.get_git_dir(), 'paper', 'figures',
        '{}_{}_{}_vs_compression'.format(embedtype, evaltype, y_metric)))
    csv_file = output_file_str + '.csv'
    plot_file = output_file_str + '.pdf'

    var_info = ['seed',[1,2,3,4,5]]
    subset_info = {
        'evaltype':[evaltype],
        'embedtype':[embedtype]
    }
    if evaltype == 'sentiment':
        assert dataset, 'Must specify dataset for sentiment analysis plots.'
        subset_info['dataset'] = [dataset]
    if embedtype in ['glove400k','fasttext1m']:
        x_metric = 'compression-ratio'
        info_per_line = {
            'kmeans':
                {
                    'compresstype':['kmeans']
                },
            'uniform (adaptive-det)': # (adaptive-det)':
                {
                    'compresstype':['uniform'],
                    'adaptive':[True],
                    'stoch':[False],
                    'skipquant':[False]
                },
            'DCCL':
                {
                    'compresstype':['dca']
                }
        }
        if embedtype == 'glove400k':
            crs = [1,1.5,3,6,8,16,32]
            info_per_line['Dim. reduction'] = {
                'compresstype':['nocompress']
            }
        else:
            crs = [8,16,32]
    else:
        x_metric = 'memory'
        info_per_line = {}
        bitrates = [1,2,4,8,16]
        for b in bitrates:
            info_per_line['b={}'.format(b)] = {
                'bitrate':[b],
                'compresstype':['uniform'],
                'adaptive':[True],
                'stoch':[False],
                'skipquant':[False]
            }
        info_per_line['b=32'] = {
            'bitrate':[32],
            'compresstype':['nocompress'],
        }
        subset_info['embeddim'] = [25,50,100,200,400]

    plt.figure()
    plot_driver(all_results,
        subset_info,
        info_per_line,
        x_metric,
        y_metric,
        logx=True,
        title='{}: {} perf. ({}) vs. memory'.format(embedtype, evaltype, y_metric),
        var_info=var_info,
        csv_file=csv_file
    )
    # plt.show()
    # plt.ylim(70.5,74.5)
    if embedtype in ['glove400k','fasttext1m']:
        plt.xticks(crs,crs)
    plt.savefig(plot_file)
    plt.close()

def plot_qa_results():
    embedtypes = ['glove400k','fasttext1m','glove-wiki400k-am']
    evaltype = 'qa'
    y_metric = 'best-f1'
    for embedtype in embedtypes:
        plot_ICML_results(embedtype, evaltype, y_metric)

def plot_sentiment_results():
    embedtypes = ['glove400k','fasttext1m','glove-wiki400k-am']
    evaltype = 'sentiment'
    y_metrics = ['val-acc','test-acc']
    datasets = ['mr','subj','cr','sst','trec','mpqa']
    for embedtype in embedtypes:
        for y_metric in y_metrics:
            for dataset in datasets:
                plot_ICML_results(embedtype, evaltype, y_metric, dataset=dataset)

def plot_intrinsic_results():
    embedtypes = ['glove400k','fasttext1m','glove-wiki400k-am']
    evaltype = 'intrinsics'
    y_metrics = ['bruni_men',
                 'luong_rare',
                 'radinsky_mturk',
                 'simlex999',
                 'ws353',
                 'ws353_relatedness',
                 'ws353_similarity',
                 'google-add',
                 'google-mul',
                 'msr-add',
                 'msr-mul',
                 'analogy-avg-score',
                 'similarity-avg-score']
    for embedtype in embedtypes:
        for y_metric in y_metrics:
            plot_ICML_results(embedtype, evaltype, y_metric)

def plot_synthetic_results():
    embedtypes = ['glove400k','fasttext1m','glove-wiki400k-am']
    evaltype = 'synthetics'
    y_metrics = ['embed-frob-error', 'embed-spec-error', 'semantic-dist', 'gram-frob-error', 'gram-spec-error']
    for embedtype in embedtypes:
        for y_metric in y_metrics:
            plot_ICML_results(embedtype, evaltype, y_metric)

def plot_embedding_standard_deviation():
    embedding_paths = [
        '/proj/smallfry/base_embeddings/fasttext1m/wiki-news-300d-1M.vec',
        '/proj/smallfry/base_embeddings/glove400k/glove.6B.50d.txt',
        '/proj/smallfry/base_embeddings/glove400k/glove.6B.100d.txt',
        '/proj/smallfry/base_embeddings/glove400k/glove.6B.200d.txt',
        '/proj/smallfry/base_embeddings/glove400k/glove.6B.300d.txt',
        '/proj/smallfry/base_embeddings/glove-wiki400k-am/2018-12-18-trainGlove/embedtype,glove_corpus,wiki400k_embeddim,25_threads,72/rungroup,2018-12-18-trainGlove_embedtype,glove_corpus,wiki400k_embeddim,25_threads,72_embeds.txt',
        '/proj/smallfry/base_embeddings/glove-wiki400k-am/2018-12-18-trainGlove/embedtype,glove_corpus,wiki400k_embeddim,50_threads,72/rungroup,2018-12-18-trainGlove_embedtype,glove_corpus,wiki400k_embeddim,50_threads,72_embeds.txt',
        '/proj/smallfry/base_embeddings/glove-wiki400k-am/2018-12-18-trainGlove/embedtype,glove_corpus,wiki400k_embeddim,100_threads,72/rungroup,2018-12-18-trainGlove_embedtype,glove_corpus,wiki400k_embeddim,100_threads,72_embeds.txt',
        '/proj/smallfry/base_embeddings/glove-wiki400k-am/2018-12-18-trainGlove/embedtype,glove_corpus,wiki400k_embeddim,200_threads,72/rungroup,2018-12-18-trainGlove_embedtype,glove_corpus,wiki400k_embeddim,200_threads,72_embeds.txt',
        '/proj/smallfry/base_embeddings/glove-wiki400k-am/2018-12-18-trainGlove/embedtype,glove_corpus,wiki400k_embeddim,400_threads,72/rungroup,2018-12-18-trainGlove_embedtype,glove_corpus,wiki400k_embeddim,400_threads,72_embeds.txt',
        '/proj/smallfry/base_embeddings/glove-wiki400k-am/2018-12-18-trainGlove/embedtype,glove_corpus,wiki400k_embeddim,800_threads,72_lr,0.025/rungroup,2018-12-18-trainGlove_embedtype,glove_corpus,wiki400k_embeddim,800_threads,72_lr,0.025_embeds.txt',
    ]
    embedtypes = ['fasttext1m'] * 1 + ['glove400k'] * 4 + ['glove-wiki400k-am'] * 6
    glove_dims = np.array([25,50,100,200,400,800])
    glove_stds = np.array([0]*6)
    ind = 0
    output_file_str = str(pathlib.PurePath(utils.get_git_dir(), 'paper', 'figures',
        'glove-wiki400k-am_embed-stdev_vs_dim'))
    csv_file = output_file_str + '.csv'
    plot_file = output_file_str + '.pdf'
    with open(csv_file,'w') as f:
        for i,embedding_path in enumerate(embedding_paths):
            embedding,_ = utils.load_embeddings(embedding_path)
            embedtype = embedtypes[i]
            dim = embedding.shape[1]
            stdev = np.std(embedding)
            f.write('{},{},{},{}\n'.format(embedding_path, embedtype, dim, stdev))
            if embedtype == 'glove-wiki400k-am':
                assert dim == glove_dims[ind]
                glove_stds[ind] = stdev
                ind = ind + 1
    plt.figure()
    plt.plot(1/np.sqrt(glove_dims), glove_stds)
    plt.title('GloVe embedding matrix st-dev vs. 1/sqrt(dim)')
    plt.xlabel('1/sqrt(dim)')
    plt.ylabel('Embedding standard deviation')
    plt.savefig(plot_file)
    plt.close()

# def construct_ICML_sentiment_figure():
#     datasets = ['mr','subj','cr','sst','trec','mpqa']
#     embedtypes = ['glove400k','fasttext1m','glove-wiki400k-am']
#     table_str = ''
#     '\\includegraphics[width=0.3\\linewidth]{figures/{}_{}_test-err_vs_compression.pdf}'
#     for dataset in datasets:
#         for i,embedtype in enumerate(embedtypes):
#             table_str = 

def plot_all_ICML_results():
    plot_qa_results()
    plot_sentiment_results()
    plot_intrinsic_results()
    plot_synthetic_results()
    plot_embedding_standard_deviation()

if __name__ == '__main__':
    #plot_frob_squared_vs_bitrate()
    #plot_dca_frob_squared_vs_lr()
    #print(dca_get_best_k_lr_per_bitrate())
    #plot_2018_11_29_fiveSeeds_QA_vs_bitrate()
    #print('hello')
    #results_path = 'C:\\Users\\avnermay\\Babel_Files\\smallfry\\results\\2018-12-16-fasttextTuneDCA_all_results.json'
    #plot_dca_frob_squared_vs_lr(results_path)
    #plot_embedding_spectra()
    # plot_ICML_qa_results()
    # get_best_lr_sentiment()
    #gather_ICML_results()
    # plot_ICML_qa_results()
    # plot_all_ICML_sentiment_results()
    plot_all_ICML_results()
