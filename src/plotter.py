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
        if result['evaltype'] == 'synthetics-large-dim':
            delta1_list = result['gram-large-dim-delta1s']
            delta2_list = result['gram-large-dim-delta2s']
            for i,(delta1,delta2) in enumerate(zip(delta1_list, delta2_list)):
                result['gram-large-dim-delta1-' + str(i)] = delta1
                result['gram-large-dim-delta2-' + str(i)] = delta2
                result['gram-large-dim-delta1-' + str(i) + "-trans"] = 1.0/(1.0-delta1)
        if result['evaltype'] == 'synthetics':
            delta1_list = result['gram-delta1s']
            delta2_list = result['gram-delta2s']
            for i,(delta1,delta2) in enumerate(zip(delta1_list, delta2_list)):
                result['gram-delta1-' + str(i)] = delta1
                result['gram-delta2-' + str(i)] = delta2
                result['gram-delta1-' + str(i) + "-trans"] = 1.0/(1.0-delta1)
            delta1_list = result['cov-delta1s']
            delta2_list = result['cov-delta2s']
            for i,(delta1,delta2) in enumerate(zip(delta1_list, delta2_list)):
                result['cov-delta1-' + str(i)] = delta1
                result['cov-delta2-' + str(i)] = delta2
                result['cov-delta1-' + str(i) + "-trans"] = 1.0/(1.0-delta1)
        # if result['evaltype'] == 'synthetics-large-dim':
            # delta1_list = result['gram-delta1s']
            # delta2_list = result['gram-delta2s']
            # for i,(delta1,delta2) in enumerate(zip(delta1_list, delta2_list)):
            #     result['gram-delta1-' + str(i)] = delta1
            #     result['gram-delta2-' + str(i)] = delta2
            #     result['gram-delta1-' + str(i) + "-trans"] = 1.0/(1.0-delta1)
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
        assert type(values) == list
        if result[key] not in values: return False
    return True

# TODO: add error bar support
def plot_driver(all_results, key_values_to_match, info_per_line, x_metric, y_metric,
                logx=False, logy=False, title=None, var_info=default_var_info,
                csv_file=None, y_metric2=None, y_metric2_evaltype=None, scatter=False):
    
    if scatter:
        assert len(key_values_to_match) != 0
        assert len(key_values_to_match['embedtype']) == 1
        assert y_metric2 and y_metric2_evaltype
        # key_values_to_match['evaltype'] = ['synthetics']
        subset_x = extract_result_subset(all_results, key_values_to_match)
        key_values_to_match['evaltype'] = [y_metric2_evaltype]
        subset_y = extract_result_subset(all_results, key_values_to_match)
        lines_x = extract_x_y_foreach_line(subset_x, info_per_line, x_metric, y_metric, var_info=var_info)
        lines_y = extract_x_y_foreach_line(subset_y, info_per_line, x_metric, y_metric2, var_info=var_info)
        title = '{}: {} perf. ({}) vs. {}'.format(key_values_to_match['embedtype'][0], y_metric2_evaltype, y_metric2, y_metric)
        plot_scatter(lines_x, lines_y, y_metric, y_metric2, logx=logx, logy=logy, title=title, csv_file=csv_file)
    else:
        if len(key_values_to_match) == 0:
            subset = all_results
        else:
            subset = extract_result_subset(all_results, key_values_to_match)
# if scatter:
#     # we reuse the extract_x_y_foreach_line to extract values
#     subset['evaltype'] = 'synthetics'
#     lines_x = extract_x_y_foreach_line(subset, info_per_line, x_metric, y_metric, var_info=var_info)
#     subset['evaltype'] = 'qa'
#     lines_y = extract_x_y_foreach_line(subset, info_per_line, x_metric, y_metric='best-f1', var_info=var_info)
#     evaltype = 'qa'
#     title = '{}: {} perf. (best-f1) vs. {}'.format(embedtype, evaltype, y_metric)
#     plot_scatter(lines_x, lines_y, x_metric, y_metric, logx=logx, logy=logy, title=title, csv_file=csv_file)
# else:
        lines = extract_x_y_foreach_line(subset, info_per_line, x_metric, y_metric, var_info=var_info)
        plot_lines(lines, x_metric, y_metric, logx=logx, logy=logy, title=title, csv_file=csv_file)


# lines_x, y contains values for x and y in the scatter plot
def plot_scatter(lines_x, lines_y, x_metric, y_metric, logx=False, logy=False, title=None, csv_file=None):
    print("scatter function")
    f = None
    if csv_file:
        f = open(csv_file,'w+')
    legend = []
    ax = plt.gcf().add_subplot(111)
    for (line_name_x,xy_x), (line_name_y,xy_y) in zip(lines_x.items(), lines_y.items()):
        assert line_name_x == line_name_y
        legend.append(line_name_x)
        # make sure the data points has the same order in lines_x
        np.testing.assert_array_equal(xy_x[0], xy_y[0])
        x_array = xy_x[1]
        y_array = xy_y[1]
        ax.scatter(x_array.ravel(), y_array.ravel())

    plt.legend(legend)
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    if logx: plt.xscale('log')
    if logy: plt.yscale('log')
    if title:
        plt.title(title)
    else:
        plt.title('{} vs {}'.format(y_metric, x_metric))


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
            '*evaltype,intrinsics*final.json', '*evaltype,synthetics_*final.json',
            '*2019-01-20-eval-*evaltype,synthetics-large-dim*final.json']
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


def plot_ICML_results(embedtype, evaltype, y_metric, dataset=None, y_metric2=None, y_metric2_evaltype=None, scatter=False, logx=False):
    # load and clean all results
    results_file = str(pathlib.PurePath(utils.get_base_dir(), 'results', 'ICML_results.json'))
    all_results = utils.load_from_json(results_file)
    all_results = clean_results(all_results)

    var_info = ['seed',[1,2,3,4,5]]
    subset_info = {
        'evaltype':[evaltype],
        'embedtype':[embedtype]
    }
    xcale_str = 'logx' if logx else 'linx'
    if evaltype == 'sentiment':
        assert dataset, 'Must specify dataset for sentiment analysis plots.'
        subset_info['dataset'] = [dataset]
        output_file_str = str(pathlib.PurePath(utils.get_git_dir(), 'paper', 'figures',
            '{}_{}_{}_{}_vs_compression_{}'.format(embedtype, evaltype, dataset, y_metric, xcale_str)))
    else:
        output_file_str = str(pathlib.PurePath(utils.get_git_dir(), 'paper', 'figures',
            '{}_{}_{}_vs_compression_{}'.format(embedtype, evaltype, y_metric, xcale_str)))
    if scatter:
        output_file_str = str(pathlib.PurePath(utils.get_git_dir(), 'paper', 'figures',
            '{}_{}_{}_vs_{}_{}'.format(embedtype, y_metric2_evaltype, y_metric2, y_metric, xcale_str)))

    # prepare filenames of output csv and pdf files.
    csv_file = output_file_str + '.csv'
    plot_file = output_file_str + '.pdf'

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
        logx=logx,
        title='{}: {} perf. ({}) vs. memory'.format(embedtype, evaltype, y_metric),
        var_info=var_info,
        csv_file=csv_file,
        y_metric2=y_metric2,
        y_metric2_evaltype=y_metric2_evaltype,
        scatter=scatter
    )
    # plt.show()
    # plt.ylim(70.5,74.5)
    if embedtype in ['glove400k','fasttext1m'] and not scatter:
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

def plot_metric_vs_performance():
    # embedtypes = ['fasttext1m']
    # evaltype = 'synthetics' # this is used in clean results
    
    # y_metrics = ['gram-frob-error', 
    #              'gram-delta1-0', 'gram-delta1-1', 'gram-delta1-2', 'gram-delta1-3', 'gram-delta1-4',
    #              'gram-delta1-0-trans', 'gram-delta1-1-trans', 'gram-delta1-2-trans', 'gram-delta1-3-trans', 'gram-delta1-4-trans']

    # GRAM DELTA PLOTS: SYNTHETICS-LARGE-DIM    
    # embedtypes = ['glove400k']
    # evaltype = 'synthetics-large-dim' # this is used in clean results
    # y_metrics = ['gram-large-dim-frob-error', 'gram-large-dim-delta1-0', 'gram-large-dim-delta1-1', 'gram-large-dim-delta1-2', 'gram-large-dim-delta1-3', 'gram-large-dim-delta1-4',
    #              'gram-large-dim-delta1-0-trans', 'gram-large-dim-delta1-1-trans', 'gram-large-dim-delta1-2-trans', 'gram-large-dim-delta1-3-trans', 'gram-large-dim-delta1-4-trans']

    # GRAM DELTA PLOTS: SYNTHETICS-LARGE-DIM
    embedtypes = ['glove-wiki400k-am']
    evaltype = 'synthetics-large-dim'
    y_metrics = ['gram-large-dim-frob-error', 'subspace-dist', 'subspace-largest-angle',
                 'gram-large-dim-delta1-0', 'gram-large-dim-delta1-1', 'gram-large-dim-delta1-2', 'gram-large-dim-delta1-3', 'gram-large-dim-delta1-4', 'gram-large-dim-delta1-5', 'gram-large-dim-delta1-6',
                 'gram-large-dim-delta1-0-trans', 'gram-large-dim-delta1-1-trans', 'gram-large-dim-delta1-2-trans', 'gram-large-dim-delta1-3-trans', 'gram-large-dim-delta1-4-trans', 'gram-large-dim-delta1-5-trans', 'gram-large-dim-delta1-6-trans']
    y_metric2 = 'best-f1'
    y_metric2_evaltype = 'qa'
    for embedtype in embedtypes:
        for y_metric in y_metrics:
            plot_ICML_results(embedtype, evaltype, y_metric, y_metric2=y_metric2, y_metric2_evaltype=y_metric2_evaltype, scatter=True)

def plot_theorem3_tighter_bound():
    dims = [300,300,200,100,50]
    embedtypes = ['fasttext1m'] + ['glove400k'] * 4
    # dims = [300]
    # embedtypes = ['glove400k']
    plt.figure()
    for i,dim in enumerate(dims):
        embedtype = embedtypes[i]
        embedpath,_ = utils.get_base_embed_info(embedtype,dim)
        base_embeds,_ = utils.load_embeddings(embedpath)
        base_sing_vals = np.linalg.svd(base_embeds, compute_uv=False)
        base_eigs = base_sing_vals**2
        eig_min = base_eigs[-1]
        scaled_eigs = base_eigs/eig_min
        n = 50
        a_list = np.logspace(-4,4, num=n)
        factors = np.zeros(n)
        for j,a in enumerate(a_list):
            factors[j] = np.average(a**2/(scaled_eigs + a)**2)   
        plt.subplot(511 + i)
        plt.plot(a_list, factors)
        plt.title('{}: dim = {}'.format(embedtype, dim))
        plt.xscale('log')
    plt.show()

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
    # plot_ICML_qa_results()
    # plot_all_ICML_sentiment_results()
    # plot_all_ICML_results()
    # plot_metric_vs_performance()
    # plot_theorem3_tighter_bound()
    # gather_ICML_results()
    plot_metric_vs_performance()
