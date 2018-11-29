import glob
import numpy as np
from matplotlib import pyplot as plt
import utils

# Returns a list of result dictionaries whose filenames match the path_regex.
def gather_results(path_regex):
    file_list = glob.glob(path_regex)
    all_results = []
    for f in file_list:
        result = utils.load_dict_from_json(f)
        all_results.append(flatten_dict(result))
    return all_results

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
                logx=False, logy=False, title=None):
    subset = extract_result_subset(all_results, key_values_to_match)
    lines = extract_x_y_foreach_line(subset, info_per_line, x_metric, y_metric)
    plot_lines(lines, x_metric, y_metric, logx=logx, logy=logy, title=title)

# lines is a dictionary of {line_name:(x,y)} pairs, where x and y are numpy
# arrays with the x and y values to be plotted.
def plot_lines(lines, x_metric, y_metric, logx=False, logy=False, title=None):
    legend = []
    for line_name,xy in lines.items():
        legend.append(line_name)
        plt.plot(xy[0],xy[1],'o--')
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
def extract_x_y_foreach_line(results, info_per_line, x_metric, y_metric):
    lines = {}
    for line_name,key_values in info_per_line.items():
        line_subset = extract_result_subset(results, key_values)
        lines[line_name] = get_x_y_values(line_subset, x_metric, y_metric)
    return lines

# extracts x,y arrays for a specific line_subset
def get_x_y_values(line_subset, x_metric, y_metric):
    x = []
    y = []
    for result in line_subset:
        x.append(result[x_metric])
        y.append(result[y_metric])
    # convert to numpy and sort results by their x values
    x,y = np.array(x), np.array(y)
    ind = np.argsort(x)
    return x[ind], y[ind]

def plot_frob_squared_vs_bitrate():
    all_results = gather_results('C:\\Users\\avnermay\\Babel_Files\\smallfry\\embeddings\\glove400k\\all_results\\*final.json')
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


def plot_dca_frob_squared_vs_lr():
    all_results = gather_results('C:\\Users\\avnermay\\Babel_Files\\smallfry\\embeddings\\glove400k\\all_results\\*final.json')
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
            title='bitrate = {}, lr vs. frob'.format(b)
        )
    plt.show()

def dca_get_best_k_lr_per_bitrate():
    all_results = gather_results('C:\\Users\\avnermay\\Babel_Files\\smallfry\\embeddings\\glove400k\\all_results\\*final.json')
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

if __name__ == '__main__':
    # plot_frob_squared_vs_bitrate()
    #plot_dca_frob_squared_vs_lr()
    print(dca_get_best_k_lr_per_bitrate())