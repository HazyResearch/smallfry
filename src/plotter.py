import glob
import pathlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import utils

default_var_info = ['gitdiff',['']]

def save_plot(filename):
    plot_dir = str(pathlib.PurePath(utils.get_git_dir(), 'paper','figures'))
    plot_filename = str(pathlib.PurePath(plot_dir, filename))
    plt.savefig(plot_filename)

# Returns a list of result dictionaries whose filenames match the path_regex.
def gather_results(path_regex):
    file_list = glob.glob(path_regex)
    return [utils.load_from_json(f) for f in file_list]

def clean_results(results):
    cleaned = []
    for result in results:
        result = flatten_dict(result)
        if result['compresstype'] == 'nocompress':
            result['bitrate'] = (32.0/300.0) * result['embeddim']
        result['compression-ratio'] = 32.0/result['bitrate']
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

def plot_2018_11_29_fiveSeeds_QA_vs_bitrate():
    results_file = str(pathlib.PurePath(utils.get_base_dir(), 'results',
                       '2018-11-29-fiveSeeds_QA_all_results.json'))
    csv_file = str(pathlib.PurePath(utils.get_base_dir(), 'results',
                       'avner_drqa_all_results.csv'))
    all_results = utils.load_from_json(results_file)
    all_results = clean_results(all_results)

    var_info = ['seed',[1,2,3,4,5]]
    plt.figure(1)
    # for seed in [1,2,3,4,5]:
    #     plt.subplot(150 + seed)
    plot_driver(all_results, {'compresstype':['kmeans','uniform','dca','nocompress'], 'seed':[1,2,3,4,5]},
        {
        'kmeans':
            {
                'compresstype':['kmeans']
            },
        # 'uniform (adaptive-stoch)':
        #     {
        #         'compresstype':['uniform'],
        #         'adaptive':[True],
        #         'stoch':[True],
        #         'skipquant':[False]
        #     },
        'uniform (adaptive-det)': # (adaptive-det)':
            {
                'compresstype':['uniform'],
                'adaptive':[True],
                'stoch':[False],
                'skipquant':[False]
            },
        # 'uniform (adaptive-skipquant)':
        #     {
        #         'compresstype':['uniform'],
        #         'adaptive':[True],
        #         'stoch':[False],
        #         'skipquant':[True]
        #     },
        # 'uniform (non-adaptive, det)':
        #     {
        #         'compresstype':['uniform'],
        #         'adaptive':[False],
        #         'stoch':[False],
        #         'skipquant':[False]
        #     },
        'DCCL':
            {
                'compresstype':['dca']
            },
        'Dim. reduction':
            {
                'compresstype':['nocompress']
            }
        },
        'compression-ratio',
        'best-f1',
        logx=True,
        title='GloVe: DrQA Perf. (F1) vs. compression ratio',
        var_info=var_info,
        csv_file=csv_file
    )
    plt.ylim(70.5,74.5)
    crs = [1,1.5,3,6,8,16,32]
    plt.xticks(crs,crs)
    save_plot('glove400k_drqa_vs_compression.pdf')

def plot_embedding_spectra():
    path = str(pathlib.PurePath(utils.get_base_dir(), 'base_embeddings',
               'glove400k', 'glove.6B.{}d.txt'))
    ds = [50,100,200,300]
    for d in ds:
        emb,_ = utils.load_embeddings(path.format(d))
        s = np.linalg.svd(emb,compute_uv=False,full_matrices=False)    
        plt.plot(s)
    plt.title('Glove400k spectra')
    plt.yscale('log')
    plt.ylabel('Singular values')
    plt.legend(['d=' + str(d) for d in ds])
    save_plot('glove400k_spectra.pdf')

    plt.figure(2)
    path = str(pathlib.PurePath(utils.get_base_dir(), 'base_embeddings',
               'fasttext1m', 'wiki-news-300d-1M.vec'))
    emb,_ = utils.load_embeddings(path)
    s = np.linalg.svd(emb,compute_uv=False,full_matrices=False)
    plt.title('fasttext1m, d=300')
    plt.plot(s)
    plt.yscale('log')
    plt.ylabel('Singular values')
    save_plot('fasttext1m_spectra.pdf')

def gather_ICML_qa_results():
    path_regexes = ['/proj/smallfry/embeddings/glove400k/2018-11-29-fiveSeeds/*/*qa_final.json',
                    '/proj/smallfry/embeddings/glove-wiki400k-am/2018-12-19-dimVsPrec/*/*qa_final.json',
                    '/proj/smallfry/embeddings/fasttext1m/2018-12-19-fiveSeeds/*/*qa_final.json']
    filenames = ['glove400k_2018-11-29-fiveSeeds',
                 'glove-wiki400k-am_2018-12-19-dimVsPrec',
                 'fasttext1m_2018-12-19-fiveSeeds']
    result_dir = '/proj/smallfry/results/'
    for i in range(len(filenames)):
        path_regex = path_regexes[i]
        filename = filenames[i]
        results = gather_results(path_regex)
        utils.save_to_json(results, result_dir + filename)

if __name__ == '__main__':
    #plot_frob_squared_vs_bitrate()
    #plot_dca_frob_squared_vs_lr()
    #print(dca_get_best_k_lr_per_bitrate())
    plot_2018_11_29_fiveSeeds_QA_vs_bitrate()
    #print('hello')
    #results_path = 'C:\\Users\\avnermay\\Babel_Files\\smallfry\\results\\2018-12-16-fasttextTuneDCA_all_results.json'
    #plot_dca_frob_squared_vs_lr(results_path)
    #plot_embedding_spectra()