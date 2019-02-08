import glob
import pathlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import re
import utils
from latexifypaper import *
from scipy.stats import spearmanr
import json
import _pickle as cp
# import warnings
# warnings.filterwarnings('ignore')

default_var_info = ['gitdiff',['']]

# setup latexify papre
default_latexify_config = {
    'aspect_ratio': [3.3, 2.8],
    'legend_frame_alpha': 0.25,
    'xlim': [1,32],
    'ylim': [None,None],
    'xtick_pos': [1,2,4,8,16,32],
    'xtick_label': [1,2,4,8,16,32],
    'xlabel': 'Compression rate',
    'logx': True,
    'minor_tick_off': True
}

# Returns a list of result dictionaries whose filenames match the path_regex.
def gather_results(path_regex):
    file_list = glob.glob(path_regex)
    return [utils.load_from_json(f) for f in file_list]

def clean_results(results):
    cleaned = []
    for result in results:
        result = flatten_dict(result)
        if 'evaltype' in result:
            result = clean_eval_result(result)
        elif 'compresstype' in result:
            result = clean_compress_result(result)
        if 'test-err' in result:
            result['test-acc'] = 1-result['test-err']
            result['val-acc'] = 1-result['val-err']
        cleaned.append(result)
    return cleaned

def clean_eval_result(result):
    if result['evaltype'] in ['synthetics-large-dim','synthetics']:
        if result['evaltype'] == 'synthetics-large-dim':
            matrix_types = ['gram']
            delta_str = 'large-dim-delta'
        else:
            assert result['evaltype'] == 'synthetics'
            matrix_types = ['gram','cov']
            delta_str = 'delta'
        for matrix_type in matrix_types:
            delta1_list = result['{}-{}1s'.format(matrix_type,delta_str)]
            delta2_list = result['{}-{}2s'.format(matrix_type,delta_str)]
            for i,(delta1,delta2) in enumerate(zip(delta1_list, delta2_list)):
                # e.g., gram-delta1-1
                result['{}-{}1-{}'.format(matrix_type,delta_str, i)] = delta1
                result['{}-{}1-{}-trans'.format(matrix_type,delta_str, i)] = 1.0/(1.0-delta1)
                result['{}-{}2-{}'.format(matrix_type,delta_str, i)] = delta2
    # FIX SUBSPACE-DIST (Compute d + k - 2 ||U^T V||_F^2)
    # large_dim = get_large_dim(result['embedtype'])
    # eig_overlap = large_dim - result['subspace-dist']
    # new_dist = large_dim + result['embeddim'] - 2 * eig_overlap
    # result['subspace-eig-distance'] = new_dist
    # result['subspace-eig-overlap'] = eig_overlap
    return result

def clean_compress_result(result):
    large_dim = get_large_dim(result['embedtype'])
    vocab = utils.get_embedding_vocab(result['embedtype'])
    if result['compresstype'] == 'nocompress':
        result['compression-ratio'] = large_dim / result['embeddim']
        result['memory'] = vocab * result['embeddim'] * 32
    elif result['compresstype'] == 'pca':
        result['compression-ratio'] = large_dim / result['pcadim']
        result['memory'] = vocab * result['pcadim'] * 32
    else:
        result['compression-ratio'] = 32.0/result['bitrate']
        result['memory'] = vocab * result['embeddim'] * result['bitrate']
    return result

def get_large_dim(embedtype):
    return 400 if embedtype == 'glove-wiki400k-am' else 300

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
        if (key not in result) or (result[key] not in values): return False
    return True


# TODO: add error bar support
def plot_driver(all_results, key_values_to_match, info_per_line, x_metric, y_metric,
                logx=False, logy=False, title=None, var_info=default_var_info,
                csv_file=None, y_metric2=None, y_metric2_evaltype=None, y_metric2_dataset=None,
                scatter=False, latexify_config=default_latexify_config):
    if scatter:
        assert len(key_values_to_match) != 0
        assert len(key_values_to_match['embedtype']) == 1
        assert y_metric2 and y_metric2_evaltype
        # key_values_to_match['evaltype'] = ['synthetics']
        subset_x = extract_result_subset(all_results, key_values_to_match)
        key_values_to_match['evaltype'] = [y_metric2_evaltype]
        if y_metric2_dataset:
            key_values_to_match['dataset'] = [y_metric2_dataset]
        subset_y = extract_result_subset(all_results, key_values_to_match)
        # print(key_values_to_match)
        lines_x = extract_x_y_foreach_line(subset_x, info_per_line, x_metric, y_metric, var_info=var_info)
        # print(lines_x)
        lines_y = extract_x_y_foreach_line(subset_y, info_per_line, x_metric, y_metric2, var_info=var_info)
        # print(lines_y)
        title = '{}: {} perf. ({}) vs. {}'.format(key_values_to_match['embedtype'][0], y_metric2_evaltype, y_metric2, y_metric)
        # return spearman rank correlation
        return plot_scatter(lines_x, lines_y, y_metric, y_metric2, logx=logx, logy=logy, title=title, csv_file=csv_file, x_normalizer=latexify_config['x_normalizer'])
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
        return None


def get_legend_name_map():
    legend_name_map = {
        'kmeans': 'K-means',
        'uniform (adaptive-det)': 'Uniform',
        'DCCL' : 'DCCL',
        'Dim. reduction': 'Dim. reduction',
    }
    for i in [1, 2, 4, 8, 16, 32]:
        legend_name_map['b={}'.format(str(i))] = '$b={}$'.format(str(i))
    return legend_name_map

# def get_label_name_map():
#     label_name_map = {
#         'best-f1': 'F1 Score',
#         'compression-ratio': 'Compression Rate',
#     }

def get_embedtype_name_map():
    embedtype_name_map = {
        'glove400k': "GloVe (Wiki'14)",
        'fasttext1m': 'fastText',
        'glove-wiki400k-am': "GloVe (Wiki'17)"
    }
    return embedtype_name_map


# lines_x, y contains values for x and y in the scatter plot
def plot_scatter(lines_x, lines_y, x_metric, y_metric, logx=False, logy=False, title=None, csv_file=None, x_normalizer=1.0):
    # print('scatter function')
    legend_name_map = get_legend_name_map()
    f = None
    if csv_file:
        f = open(csv_file,'w+')
    legend = []
    ax = plt.gcf().add_subplot(111)

    full_x_list = []
    full_y_list = []
    for (line_name_x,xy_x), (line_name_y,xy_y) in zip(lines_x.items(), lines_y.items()):
        assert line_name_x == line_name_y
        legend.append(legend_name_map[line_name_x])
        # make sure the data points has the same order in lines_x
        np.testing.assert_array_equal(xy_x[0], xy_y[0])
        x_array = xy_x[1]
        if x_metric == 'subspace-eig-overlap':
            x_array = 1.0-x_array / float(x_normalizer)
        y_array = xy_y[1]
        ax.scatter(x_array.ravel(), y_array.ravel())
        full_x_list += x_array.ravel().tolist()
        full_y_list += y_array.ravel().tolist()

    print('spearman rank correlation\n', spearmanr(np.array(full_x_list), np.array(full_y_list)), '\n')
    print(len(full_x_list), len(full_y_list), np.array(full_x_list), np.array(full_y_list), np.unique(full_x_list), np.unique(full_y_list))
    input('Press Enter to continue...')

    plt.legend(legend)
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    if logx: plt.xscale('log')
    if logy: plt.yscale('log')
    if title:
        plt.title(title)
    else:
        plt.title('{} vs {}'.format(y_metric, x_metric))
    return spearmanr(np.array(full_x_list), np.array(full_y_list))


# lines is a dictionary of {line_name:(x,y)} pairs, where x and y are numpy
# arrays with the x and y values to be plotted.
def plot_lines(lines, x_metric, y_metric, logx=False, logy=False, title=None, csv_file=None):
    legend_name_map = get_legend_name_map()    
    f = None
    if csv_file:
        f = open(csv_file,'w+')
    legend = []
    for line_name,xy in lines.items():
        legend.append(legend_name_map[line_name])
        # plt.plot(xy[0],xy[1],'o--')
        sorted_x = xy[0]
        y_array = xy[1]
        y_avg = np.average(y_array,axis=0)
        y_std = np.std(y_array,axis=0)
        # if line_name == 'Dim. reduction':
        #     plt.errorbar(sorted_x, y_avg, yerr=y_std, marker='o', capthick=4, capsize=10000)
        # else:
        plt.errorbar(sorted_x, y_avg, yerr=y_std, marker='o', markersize=5, capthick=2, capsize=5)
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
            '*2019-01-20-eval-*evaltype,synthetics-large-dim*final.json',
            '*2019-02-0*-eval-*evaltype,synthetics-large-dim*final.json']
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
    path_regex = '/proj/smallfry/embeddings/*/*/*/*evaltype,sent*tunelr,True*final.json'
    all_results = clean_results(gather_results(path_regex))
    # first gather list of base_embeds
    base_embeds = []
    for result in all_results:
        if result['base-embed-path'] not in base_embeds:
            base_embeds.append(result['base-embed-path'])
        if ('compresstype,pca_seed,1_' in result['compressed-embed-path'] and
            result['compressed-embed-path'] not in base_embeds):
            # For PCA embeddings, we treat the seed=1 embedding as the 'base_embed'
            # This way, the best LR is chosen per PCA dimension.
            base_embeds.append(result['compressed-embed-path'])
    assert len(base_embeds) == 15
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
            if 'compresstype,pca_seed,1_' in base_embed:
                # m = re.search('_rungroup,(.+?)_(.*)_pcadim,(.+?)_', base_embed)
                m = re.search('_pcadim,(.+?)/', base_embed)
                assert m, 'Improper base_embed path'
                pcadim = int(m.group(1))
                # TODO: Need to make this more specific if add more PCA results
                base_embed_results = extract_result_subset(all_results,
                    {'pcadim':[pcadim], 'dataset':[dataset]})
            else:
                base_embed_results = extract_result_subset(all_results,
                    {'base-embed-path':[base_embed], 'dataset':[dataset], 'compresstype':['nocompress']})
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

def latexify_setup_fig(config=None):
        latexify(columns=1)
        if ('aspect_ratio' in config.keys()) and (config['aspect_ratio'] is not None):
            plt.figure(figsize=config['aspect_ratio'])
        else:
            plt.figure()
        ax = plt.subplot(111)
        return ax

def latexify_finalize_fig(ax, config=None):
    plt.grid()
    leg = plt.gca().legend_
    leg.get_frame().set_linewidth(0.0)
    if 'logx' in config.keys() and (config['logx'] is not None):
        if config['logx']:
            plt.xscale('log')
        else:
            plt.xscale('linear')
    if 'legend_frame_alpha' in config.keys() and (config['legend_frame_alpha'] is not None):
        leg.framealpha = config['legend_frame_alpha']
    if 'xlim' in config.keys() and (config['xlim'] is not None):
        plt.xlim(config['xlim'])
    if 'ylim' in config.keys() and (config['ylim'] is not None):
        plt.ylim(config['ylim'])
    if 'xlabel' in config.keys() and (config['xlabel'] is not None):
        plt.xlabel(config['xlabel'])
    if 'ylabel' in config.keys() and (config['ylabel'] is not None):
        plt.ylabel(config['ylabel'])
    if 'title' in config.keys() and (config['title'] is not None):
        plt.title(config['title'])
    if 'xtick_pos' in config.keys() and (config['xtick_pos'] is not None):
        plt.xticks(config['xtick_pos'], config['xtick_label'])
    if 'minor_tick_off' in config.keys() and (config['minor_tick_off'] is not None):
        if config['minor_tick_off']:
            plt.minorticks_off()
    format_axes(ax)
    plt.tight_layout()


spearman_dict = {}

def plot_ICML_results(embedtype, evaltype, y_metric, dataset=None,
                      y_metric2=None, y_metric2_evaltype=None, scatter=False, 
                      logx=False, latexify_config=default_latexify_config):
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
    if scatter:
        if y_metric2_evaltype == 'sentiment':
            output_file_str = str(pathlib.PurePath(utils.get_git_dir(), 'paper', 'figures',
                '{}_{}_{}_{}_vs_{}_{}'.format(embedtype, y_metric2_evaltype, dataset, y_metric2, y_metric, xcale_str)))
        else:
            output_file_str = str(pathlib.PurePath(utils.get_git_dir(), 'paper', 'figures',
                '{}_{}_{}_vs_{}_{}'.format(embedtype, y_metric2_evaltype, y_metric2, y_metric, xcale_str)))
    else:
        if evaltype == 'sentiment':
            assert dataset, 'Must specify dataset for sentiment analysis plots.'
            subset_info['dataset'] = [dataset]
            output_file_str = str(pathlib.PurePath(utils.get_git_dir(), 'paper', 'figures',
                '{}_{}_{}_{}_vs_compression_{}'.format(embedtype, evaltype, dataset, y_metric, xcale_str)))
        else:
            output_file_str = str(pathlib.PurePath(utils.get_git_dir(), 'paper', 'figures',
                '{}_{}_{}_vs_compression_{}'.format(embedtype, evaltype, y_metric, xcale_str)))

    # prepare filenames of output csv and pdf files.
    csv_file = output_file_str + '.csv'
    plot_file = output_file_str + '.pdf'

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
    if embedtype == 'glove-wiki400k-am':
        crs = [1,2,4,8,16,32]
        info_per_line['Dim. reduction'] = {
            'compresstype':['nocompress'],
            'embeddim': [50,100,200,400]
        }
    elif embedtype == 'glove400k':
        crs = [1,1.5,3,6,8,16,32]
        info_per_line['Dim. reduction'] = {
            'compresstype':['nocompress'],
            'embeddim': [50,100,200,300]
        }
    elif embedtype == 'fasttext1m':
        crs = [1,1.5,3,6,8,16,32]
        info_per_line['Dim. reduction'] = {
            'compresstype':['pca'],
            'pcadim': [50,100,200,300]
        }
    x_metric = 'compression-ratio'
    if y_metric == 'embed-frob-error' and scatter == True:
        subset_info['embeddim'] = [get_large_dim(embedtype)]
    ####  Code below is for plotting all the different bitrates for glove-wiki400k-am. ####
    # x_metric = 'memory'
    # info_per_line = {}
    # bitrates = [1,2,4,8,16]
    # for b in bitrates:
    #     info_per_line['b={}'.format(b)] = {
    #         'bitrate':[b],
    #         'compresstype':['uniform'],
    #         'adaptive':[True],
    #         'stoch':[False],
    #         'skipquant':[False],
    #         'embeddim':[25,50,100,200,400]
    #     }
    # info_per_line['b=32'] = {
    #     'bitrate':[32],
    #     'compresstype':['nocompress'],
    #     'embeddim':[25,50,100,200,400]
    # }

    ax = latexify_setup_fig(latexify_config)
    # plt.figure()

    print('check ', latexify_config)

    return_info = plot_driver(all_results,
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
        y_metric2_dataset=dataset,
        scatter=scatter,
        latexify_config=latexify_config
    )
    # plt.show()
    # plt.ylim(70.5,74.5)
    # if embedtype in ['glove400k','fasttext1m'] and not scatter:
    #     plt.xticks(crs,crs)
    if scatter:
        latexify_config['title'] += r', $\rho={0:.2f}$'.format(return_info[0])
        if y_metric2 == 'test-acc':
            if dataset == 'sst':
                save = True
            else:
                save = False
        else:
            save = True
        if save:
            key_name = embedtype + ', ' + y_metric2 + ', ' + y_metric
            spearman_dict[key_name] = return_info[0]
    latexify_finalize_fig(ax, latexify_config)

    print(plot_file)

    plt.savefig(plot_file)
    plt.close()

def plot_qa_results():
    embedtypes = ['glove-wiki400k-am', 'glove400k','fasttext1m',]
    evaltype = 'qa'
    y_metric = 'best-f1'
    latexify_config = default_latexify_config
    embedtype_name_map = get_embedtype_name_map()
    latexify_config['ylabel'] = 'F1 score'
    for embedtype in embedtypes:
        latexify_config['x_normalizer'] = get_large_dim(embedtype)
        latexify_config['title'] = embedtype_name_map[embedtype] + ', QA'
        plot_ICML_results(embedtype, evaltype, y_metric, latexify_config=latexify_config)

def plot_sentiment_results():
    embedtypes = ['glove400k','fasttext1m','glove-wiki400k-am']
    evaltype = 'sentiment'
    y_metrics = ['val-acc','test-acc']
    latexify_config = default_latexify_config
    embedtype_name_map = get_embedtype_name_map()
    datasets = ['mr','subj','cr','sst','trec','mpqa']
    for embedtype in embedtypes:
        latexify_config['x_normalizer'] = get_large_dim(embedtype)
        for y_metric in y_metrics:
            if y_metric == 'val-acc':
                latexify_config['ylabel'] = 'Validation acc.'
            elif y_metric == 'test-acc':
                latexify_config['ylabel'] = 'Test acc.'
            for dataset in datasets:
                latexify_config['title'] = embedtype_name_map[embedtype] + ', sentiment'
                plot_ICML_results(embedtype, evaltype, y_metric, dataset=dataset, latexify_config=latexify_config)

def plot_intrinsic_results():
    embedtypes = ['glove400k','fasttext1m','glove-wiki400k-am']
    evaltype = 'intrinsics'
    # y_metrics = ['bruni_men',
    #              'luong_rare',
    #              'radinsky_mturk',
    #              'simlex999',
    #              'ws353',
    #              'ws353_relatedness',
    #              'ws353_similarity',
    #              'google-add',
    #              'google-mul',
    #              'msr-add',
    #              'msr-mul',
    #              'analogy-avg-score',
    #              'similarity-avg-score']
    y_metrics = [r'analogy-avg-score',
                 r'similarity-avg-score']
    latexify_config = default_latexify_config
    embedtype_name_map = get_embedtype_name_map()
    for embedtype in embedtypes:
        latexify_config['x_normalizer'] = get_large_dim(embedtype)
        for y_metric in y_metrics:
            if y_metric == r'analogy-avg-score':
                latexify_config['title'] = embedtype_name_map[embedtype] + ', analogy'
                latexify_config['ylabel'] = 'Analogy average score'
            elif y_metric == r'similarity-avg-score':
                latexify_config['title'] = embedtype_name_map[embedtype] + ', similarity'
                latexify_config['ylabel'] = 'Similarity average score'
            plot_ICML_results(embedtype, evaltype, y_metric, latexify_config=latexify_config)

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

def plot_metric_vs_performance(y_metric2_evaltype, use_large_dim, logx):
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

    # embedtypes = ['glove400k', 'glove-wiki400k-am', 'fasttext1m',]
    embedtypes = ['glove-wiki400k-am']
    latexify_config = default_latexify_config
    embedtype_name_map = get_embedtype_name_map()
    # SET Y_METRIC1 PARAMS
    if use_large_dim:
        evaltype = 'synthetics-large-dim'
        only_compute_delta2 = False
        if only_compute_delta2:
            y_metric1s = ['gram-large-dim-delta2-0', 'gram-large-dim-delta2-1', 'gram-large-dim-delta2-2', 'gram-large-dim-delta2-3', 'gram-large-dim-delta2-4', 'gram-large-dim-delta2-5', 'gram-large-dim-delta2-6']
        else:
            # y_metric1s = ['embed-frob-error', 'gram-large-dim-frob-error', 'subspace-eig-overlap', 
            #         'gram-large-dim-delta1-2-trans', 
            #         'gram-large-dim-delta2-2',
            #         ]
            y_metric1s = ['embed-frob-error']

            # y_metric1s = ['gram-large-dim-frob-error', 'subspace-eig-distance', 'subspace-eig-overlap', 'subspace-largest-angle',
            #         'gram-large-dim-delta1-0', 'gram-large-dim-delta1-1', 'gram-large-dim-delta1-2', 'gram-large-dim-delta1-3', 'gram-large-dim-delta1-4', 'gram-large-dim-delta1-5', 'gram-large-dim-delta1-6',
            #         'gram-large-dim-delta1-0-trans', 'gram-large-dim-delta1-1-trans', 'gram-large-dim-delta1-2-trans', 'gram-large-dim-delta1-3-trans', 'gram-large-dim-delta1-4-trans', 'gram-large-dim-delta1-5-trans', 'gram-large-dim-delta1-6-trans',
            #         'gram-large-dim-delta2-0', 'gram-large-dim-delta2-1', 'gram-large-dim-delta2-2', 'gram-large-dim-delta2-3', 'gram-large-dim-delta2-4', 'gram-large-dim-delta2-5', 'gram-large-dim-delta2-6'
            #         ]            
    else:
        evaltype = 'synthetics'
        y_metric1s = ['embed-frob-error', 'embed-spec-error', 'embed-mean-euclidean-dist', 'semantic-dist']

    # SET Y_METRIC2 PARAMS
    if y_metric2_evaltype == 'qa':
        y_metric2s = ['best-f1']
        datasets = [None]
    elif y_metric2_evaltype == 'sentiment':
        y_metric2s = ['test-acc']
        datasets = ['mr','subj','cr','sst','trec','mpqa']
    elif y_metric2_evaltype == 'intrinsics':
        y_metric2s = ['analogy-avg-score','google-mul','google-add','msr-mul','msr-add']
        # y_metric2s = ['analogy-avg-score','similarity-avg-score','google-mul','google-add','msr-mul','msr-add']
        datasets = [None]

    # logxs = [True,False]
    for embedtype in embedtypes:
        latexify_config['xlim'] = [0,None]
        latexify_config['ylim'] = [None, None]
        latexify_config['x_normalizer'] = get_large_dim(embedtype)
        # latexify_config['xlabel'] = 'Compression rate'
        # latexify_config['logx'] = True
        latexify_config['minor_tick_off'] = True
        # latexify_config['title'] = embedtype_name_map[embedtype]
        for y_metric1 in y_metric1s:
            if y_metric1 == 'embed-frob-error':
                evaltype = 'synthetics'
            else:
                evaltype = 'synthetics-large-dim'
            if 'gram' in y_metric1 and 'frob' in y_metric1:
                latexify_config['xlabel'] = 'PIP loss'
            elif 'embed' in y_metric1 and 'frob' in y_metric1:
                latexify_config['xlabel'] = 'Embed. reconstruction. error'
            elif 'delta1' in y_metric1 and 'trans' in y_metric1:
                latexify_config['xlabel'] = r'$1/(1 - \Delta_1)$'
            elif 'delta2' in y_metric1:
                latexify_config['xlabel'] = r'$\Delta_2$'
            elif y_metric1 == 'subspace-eig-overlap':
                latexify_config['xlabel'] = r'1 - $\mathcal{E}$'
            for y_metric2 in y_metric2s:
                # # for logx in logxs:
                # if y_metric2_evaltype == 'qa':
                #     latexify_config['ylabel'] = 'F1 score'
                #     latexify_config['title'] = embedtype_name_map[embedtype] + ', QA'
                # elif y_metric2_evaltype == 'sentiment':
                #     latexify_config['ylabel'] = 'Test acc.'
                #     latexify_config['title'] = embedtype_name_map[embedtype] + ', sentiment'
                # elif y_metric2_evaltype == 'intrinsics':
                #     if y_metric2 == 'analogy-avg-score':
                #         latexify_config['ylabel'] = 'Analogy average score'
                #         latexify_config['title'] = embedtype_name_map[embedtype] + ', analogy'
                #     else:
                #         latexify_config['ylabel'] = 'Similarity average score'
                #         latexify_config['title'] = embedtype_name_map[embedtype] + ', similarity'
                for dataset in datasets:
                    # for logx in logxs:
                    if y_metric2_evaltype == 'qa':
                        latexify_config['ylabel'] = 'F1 score'
                        latexify_config['title'] = embedtype_name_map[embedtype] + ', QA'
                    elif y_metric2_evaltype == 'sentiment':
                        latexify_config['ylabel'] = 'Test acc.'
                        latexify_config['title'] = embedtype_name_map[embedtype] + ', sentiment'
                    elif y_metric2_evaltype == 'intrinsics':
                        if y_metric2 == 'analogy-avg-score':
                            latexify_config['ylabel'] = 'Analogy average score'
                            latexify_config['title'] = embedtype_name_map[embedtype] + ', analogy'
                        else:
                            latexify_config['ylabel'] = 'Similarity average score'
                            latexify_config['title'] = embedtype_name_map[embedtype] + ', similarity'

                    print('Embedtype = {}, {} vs {}, dataset = {}'.format(
                          embedtype, y_metric1, y_metric2, dataset))
                    plot_ICML_results(embedtype, evaltype, y_metric1, y_metric2=y_metric2,
                        y_metric2_evaltype=y_metric2_evaltype, scatter=True, logx=logx,
                        dataset=dataset, latexify_config=latexify_config)

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

def print_spearrank_table_blob():
    with open('./spearman_dict', 'rb') as f:
        spearman_dict = cp.load(f)
    embedtypes = ['glove400k', 'glove-wiki400k-am', 'fasttext1m',]
    x_metrics = ['embed-frob-error', 'gram-large-dim-frob-error', 
                    'gram-large-dim-delta1-2-trans', 
                    'gram-large-dim-delta2-2', 'subspace-eig-overlap']
    y_metrics = ['best-f1', 'test-acc', 'analogy-avg-score', 'similarity-avg-score', 'google-mul','google-add','msr-mul','msr-add']
    for x in x_metrics:
        info = ' '
        for y in y_metrics:
            for embed in embedtypes:
                key = embed + ', ' + y + ', ' + x
                if key in spearman_dict.keys():
                    # info += r'{0:.2f}/'.format(spearman_dict[key])
                    info += r'{0:.5f}/'.format(spearman_dict[key])
            info = info[:-1]
            info += '  &  '
        print(x, info)

if __name__ == '__main__':
    # # lines
    # plot_qa_results()
    # plot_intrinsic_results()
    # plot_sentiment_results()
    
    # #plot_frob_squared_vs_bitrate()
    # #plot_dca_frob_squared_vs_lr()
    # #print(dca_get_best_k_lr_per_bitrate())
    # #plot_2018_11_29_fiveSeeds_QA_vs_bitrate()
    # #print('hello')
    # #results_path = 'C:\\Users\\avnermay\\Babel_Files\\smallfry\\results\\2018-12-16-fasttextTuneDCA_all_results.json'
    # #plot_dca_frob_squared_vs_lr(results_path)
    # #plot_embedding_spectra()
    # # plot_ICML_qa_results()
    # # get_best_lr_sentiment()
    # # plot_ICML_qa_results()
    # # plot_all_ICML_sentiment_results()
    # # plot_all_ICML_results()
    # # plot_metric_vs_performance()
    # # plot_theorem3_tighter_bound()
    # # gather_ICML_results()

    # scatter plots
    logx = False
    # # use_large_dims = [True, False]
    use_large_dims = [True]
    for use_large_dim in use_large_dims:
        # plot_metric_vs_performance('qa', use_large_dim, logx)
        # plot_metric_vs_performance('sentiment', use_large_dim, logx)
        plot_metric_vs_performance('intrinsics', use_large_dim, logx)
    print(spearman_dict)
    with open('./spearman_dict', 'wb') as f:
        cp.dump(spearman_dict, f)

    print_spearrank_table_blob()
