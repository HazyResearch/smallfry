import argh
import pathlib
import os
import numpy as np
import maker

'''
CORE LAUNCH METHODS: launch and qsub_launch
'''
#globally stores a launch
log = []
#maker cmd creator launch path
launch_path = str(pathlib.PurePath(maker.get_launch_path(), 'maker'))
qsub_log_path = str(pathlib.PurePath(maker.get_qsub_log_path(), 'maker'))
qsub_preamble = "qsub -V -b y -wd"

def launch(method, params):
    s = ''
    maker_path = str(pathlib.PurePath(os.path.dirname(os.path.realpath(__file__)),'maker.py'))
    python36_maker_cmd = 'python %s' % maker_path
    if method == 'kmeans':
        s = '%s --method kmeans --base %s --basepath %s --seed %s --outputdir %s --rungroup %s --bitsperblock %s --blocklen %s --ibr %s' % ((python36_maker_cmd,)+params)
    elif method == 'dca':
        s = '%s --method dca --base %s --basepath %s --seed %s --outputdir %s --rungroup %s --m %s --k %s --ibr %s' % ((python36_maker_cmd,)+params)
    elif method == 'baseline' or method == 'stochround' or method == 'midriser' or method == 'optranuni':
        s = '%s --method %s --base %s --basepath %s --seed %s --outputdir %s --rungroup %s --ibr %s' % ((python36_maker_cmd,)+(method,)+params)
    else:
        raise ValueError(f"bad method name in launch: {method}")
    return s

def qsub_launch(method, params):
    return 'qsub -V -b y -wd %s %s ' % (qsub_log_path, launch(method, params))

def qsub_launch_config(config):
    s = ''
    global qsub_preamble
    maker_path = str(pathlib.PurePath(os.path.dirname(os.path.realpath(__file__)),'maker.py'))
    python_maker_cmd = 'python %s' % maker_path
    if config['method'] == 'dca':
        s = f"{python_maker_cmd} --method {config['method']} --base {config['base']} --basepath {config['basepath']} \
        --seed {config['seed']} --outputdir {config['outputdir']} --rungroup {config['rungroup']} --ibr {config['ibr']} \
        --m {config['m']} --k {config['k']}  --batchsize {config['batchsize']} --gradclip {config['gradclip']} --lr {config['lr']} --tau {config['tau']}"
        s = f"{qsub_preamble} {qsub_log_path} {s}"
    elif config['method'] == 'kmeans':
        s = f"{python_maker_cmd} --method {config['method']} --base {config['base']} --basepath {config['basepath']} \
                --seed {config['seed']} --outputdir {config['outputdir']} --rungroup {config['rungroup']} --ibr {config['ibr']} \
                --bitsperblock {config['bitsperblock']} --blocklen {config['blocklen']} --solver {config['solver']}"
        s = f"{qsub_preamble} {qsub_log_path} {s}"
    elif config['method'] == 'midriser':
        s = f"{python_maker_cmd} --method {config['method']} --base {config['base']} --basepath {config['basepath']} \
                --seed {config['seed']} --outputdir {config['outputdir']} --rungroup {config['rungroup']} --ibr {config['ibr']}"
        s = f"{qsub_preamble} {qsub_log_path} {s}"
    else:
        raise ValueError(f"bad method name in launch: {config['method']}")
    return s

'''
HELPER METHODS FOR COMMON SWEEP STYLES (and logging)
'''

def log_launch(name):
    log_launch_path = str(pathlib.PurePath( launch_path, name ))
    with open(log_launch_path, 'w+') as llp:
        llp.write('\n'.join(log))

def dca_param_gen(bitrates, base_embeds_path, upper_power=8, size_tol=0.15):
    dca_params  = []
    k_s = [2**i for i in range(1,upper_power+1)]
    base_embeds,_ = maker.load_embeddings(base_embeds_path)
    v,d = base_embeds.shape
    m = lambda k,v,d,br: int(np.round(0.125*br*v*d/(0.125*v*np.log2(k) + 4*d*k)))
    get_size_in_bits = lambda v,m,k,d: (0.125*v*m*np.log2(k) + 4*d*k*m)*8
    for k in k_s:
        for br in bitrates:
            param = ((m(k,v,d,br),k))
            if param[0] > 0 and abs(get_size_in_bits(v, param[0], param[1], d)-br*v*d) < size_tol*br*v*d:
                dca_params.append((m(k,v,d,br),k))
    return dca_params

def sweep(method, rungroup, base_embeds, base_embeds_path, seeds, params, qsub=True):       
    '''a subroutine for complete 'sweeps' of params'''
    l = qsub_launch if qsub else launch
    for seed in seeds:
        for e in range(len(base_embeds)):
            for p in params:
                cmd = l(method,(
                        base_embeds[e],
                        base_embeds_path[e],
                        seed,
                        maker.get_base_outputdir(),
                        rungroup,
                        p[0],
                        p[1],
                        p[2]))
                log.append(cmd)

def sweep_configs(configs):
    for config in configs:
        log.append(launch_config(config))

'''
LAUNCH ROUTINES BELOW THIS LINE =========================
'''

def make_exp9_10_15_18(name):
    rungroup = 'exp9-dim-vs-prec'
    method = 'glove'
    dims = [320,160,80,40,10]
    configs = []
    for dim in dims:
        config = dict()
        config['rungroup'] = rungroup
        config['method'] = method
        config['corpus'] = 'text8'
        config['dim'] = dim
        config['outputdir'] = generate.get_base_outputdir()
        config['memusage'] = 256
        config['seed'] = 1234
        configs.append(config)
    sweep_configs(configs, False)
    log_launch(generate.get_log_name(name, rungroup))

#IMPORTANT!! this line determines which cmd will be run
cmd = [make_baseline_exp7_10_11_18]

parser = argh.ArghParser()
parser.add_commands(cmd)

if __name__ == '__main__':
    parser.dispatch()
