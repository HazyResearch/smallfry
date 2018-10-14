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
def make_optranuni_exp2_10_9_18(name):
    rungroup = 'experiment2-5X-seeds'
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    ibrs = [1,2,4]
    base_embeds = ['fasttext','glove']
    base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
    base_embeds_path = [base_path_ft, base_path_glove]
    seeds = [4974]
    for seed in seeds:
        for i in [0,1]: #loop over baselines: fasttext and glove
            for ibr in ibrs:
                log.append(launch('optranuni',(base_embeds[i], base_embeds_path[i], seed, maker.get_base_outputdir(), rungroup, ibr)))
    log_launch(maker.get_log_name(name, rungroup))



def test_optranuni_exp2_10_9_18(name):
    rungroup = 'experiment2-5X-seeds-DBG'
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    ibrs = [1,2,4]
    base_embeds = ['fasttext','glove']
    base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
    base_embeds_path = [base_path_ft, base_path_glove]
    seeds = [4974]
    for seed in seeds:
        for i in [0,1]: #loop over baselines: fasttext and glove
            for ibr in ibrs:
                log.append(launch('optranuni',(base_embeds[i], base_embeds_path[i], seed, maker.get_base_outputdir(), rungroup, ibr)))
    log_launch(maker.get_log_name(name, rungroup))

def launch_official_dca_sweep2_exp5_10_8_18(name):
    rungroup = 'experiment5-dca-hp-tune'
    methods = ['dca']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    ibrs = [0.1,0.25,0.5,1,2,4]
    base_embeds = ['fasttext','glove']
    base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
    base_embeds_path = [base_path_ft, base_path_glove]
    seeds = [1234]
    lrs = [1e-4,1e-3,1e-5]
    batchsizes = [64, 128]
    taus = [1,2,0.5]
    ibr_2_mks = dict()
    ibr_2_mks[str(0.1)] = [(7,16),(5,32),(4,64),(3,128)]
    ibr_2_mks[str(0.25)] = [(23,8),(17,16),(13,32),(10,64)]
    ibr_2_mks[str(0.5)] = [(72,4),(47,8),(34,16)]
    ibr_2_mks[str(1)] = [(286,2),(143,4),(94,8)]
    ibr_2_mks[str(2)] = [(573,2),(286,4),(188,8)]
    ibr_2_mks[str(4)] = [(1145,2),(573,4),(376,8)]

    configs = []
    for seed in seeds:
        for i in [0,1]:
            for batchsize in batchsizes:
                for gradclip in [0.01]:
                    for lr in lrs:
                        for tau in taus:
                            for ibr in ibrs:
                                mks = ibr_2_mks[str(ibr)]
                                print(mks)
                                for mk in mks:
                                    print(mk)
                                    k = mk[1]
                                    if base_embeds[i] == 'glove' and ibr == 0.1 and k == 128:
                                        continue
                                    if base_embeds[i] == 'glove' and ibr == 0.25 and k == 64:
                                        continue
                                    if base_embeds[i] == 'fasttext' and ibr == 0.1 and k == 16:
                                        continue
                                    if base_embeds[i] == 'fasttext' and ibr == 0.25 and k == 8:
                                        continue

                                    config = dict()
                                    config['m'] = mk[0]
                                    config['k'] = mk[1]
                                    config['method'] = methods[0]
                                    config['ibr'] = ibr
                                    config['seed'] = seed
                                    config['outputdir'] = maker.get_base_outputdir()
                                    config['basepath'] = base_embeds_path[i]
                                    config['base'] = base_embeds[i]
                                    config['lr'] = lr
                                    config['rungroup'] = rungroup
                                    config['tau'] = tau
                                    config['gradclip'] = gradclip
                                    config['batchsize'] = batchsize
                                    configs.append(config)
        sweep_configs(configs)
        log_launch(maker.get_log_name(name, rungroup))


def launch_trial_dca_br6(name):
    rungroup = 'trial-dca-br6'
    methods = ['dca']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    ibrs = [4]
    base_embeds = ['fasttext']
    base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_embeds_path = [base_path_ft]
    seeds = [1]
    lrs = [1e-4]
    batchsizes = [64, 128]
    m = 573
    k = 4
    configs = []
    for seed in seeds:
        for batchsize in batchsizes:
            for i in [0]: #loop over baselines: fasttext and glove
                for lr in lrs:
                    for ibr in ibrs:
                        config = dict()
                        config['m'] = 573
                        config['k'] = 4
                        config['method'] = methods[0]
                        config['ibr'] = ibr
                        config['seed'] = seed
                        config['outputdir'] = maker.get_base_outputdir()
                        config['basepath'] = base_embeds_path[0]
                        config['base'] = base_embeds[0]
                        config['lr'] = lr
                        config['rungroup'] = rungroup
                        config['tau'] = 1.0
                        config['gradclip'] = 0.001
                        config['batchsize'] = batchsize
                        configs.append(config)
    sweep_configs(configs)
    log_launch(maker.get_log_name(name, rungroup))

def launch_trial_dca_sweep_br6(name):
    rungroup = 'trial-dca-sweep'
    methods = ['dca']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    ibrs = [6]
    base_embeds = ['fasttext']
    base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_embeds_path = [base_path_ft]
    seeds = [1]
    lrs = [1e-4]
    batchsizes = [64]
    mks = [(1718,2),(127,256),(312,32)]
    configs = []
    for seed in seeds:
        for batchsize in batchsizes:
            for i in [0]: #loop over baselines: fasttext and glove
                for lr in lrs:
                    for ibr in ibrs:
                        for mk in mks:
                            config = dict()
                            config['m'] = mk[0]
                            config['k'] = mk[1]
                            config['method'] = methods[0]
                            config['ibr'] = ibr
                            config['seed'] = seed
                            config['outputdir'] = maker.get_base_outputdir()
                            config['basepath'] = base_embeds_path[0]
                            config['base'] = base_embeds[0]
                            config['lr'] = lr
                            config['rungroup'] = rungroup
                            config['tau'] = 1.0
                            config['gradclip'] = 0.001
                            config['batchsize'] = batchsize
                            configs.append(config)
    sweep_configs(configs)
    log_launch(maker.get_log_name(name, rungroup))

def launch_trial_dca_sweep(name):
    rungroup = 'trial-dca-sweep'
    methods = ['dca']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    ibrs = [4]
    base_embeds = ['fasttext']
    base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_embeds_path = [base_path_ft]
    seeds = [1]
    lrs = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    batchsizes = [64, 128]
    m = 573
    k = 4
    configs = []
    for seed in seeds:
        for batchsize in batchsizes:
            for i in [0]: #loop over baselines: fasttext and glove
                for lr in lrs:
                    for ibr in ibrs:
                        config = dict()
                        config['m'] = 573
                        config['k'] = 4
                        config['method'] = methods[0]
                        config['ibr'] = ibr
                        config['seed'] = seed
                        config['outputdir'] = maker.get_base_outputdir()
                        config['basepath'] = base_embeds_path[0]
                        config['base'] = base_embeds[0]
                        config['lr'] = lr
                        config['rungroup'] = rungroup
                        config['tau'] = 1.0
                        config['gradclip'] = 0.001
                        config['batchsize'] = batchsize
                        configs.append(config)
    sweep_configs(configs)
    log_launch(maker.get_log_name(name, rungroup))

def launch_official_midriser_10_5_18(name):
    rungroup = 'experiment2-5X-seeds'
    methods = ['midriser']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    ibrs = [1,2,4]
    base_embeds = ['fasttext','glove']
    base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
    base_embeds_path = [base_path_ft, base_path_glove]
    seeds = [4974, 7737, 6665, 6117, 8559]
    for seed in seeds:
        for i in [0,1]: #loop over baselines: fasttext and glove
            for ibr in ibrs:
                log.append(qsub_launch('midriser',(base_embeds[i], base_embeds_path[i], seed, maker.get_base_outputdir(), rungroup, ibr)))
    log_launch(maker.get_log_name(name, rungroup))

def launch_official_midriser_10_5_18(name):
    rungroup = 'experiment2-5X-seeds'
    methods = ['midriser']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    ibrs = [1,2,4]
    base_embeds = ['fasttext','glove']
    base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
    base_embeds_path = [base_path_ft, base_path_glove]
    seeds = [4974, 7737, 6665, 6117, 8559]
    for seed in seeds:
        for i in [0,1]: #loop over baselines: fasttext and glove
            for ibr in ibrs:
                log.append(qsub_launch('midriser',(base_embeds[i], base_embeds_path[i], seed, maker.get_base_outputdir(), rungroup, ibr)))
    log_launch(maker.get_log_name(name, rungroup))

def launch_official_midriser_10_5_18(name):
    rungroup = 'experiment2-5X-seeds'
    methods = ['midriser']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    ibrs = [1,2,4,6]
    base_embeds = ['fasttext','glove']
    base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
    base_embeds_path = [base_path_ft, base_path_glove]
    seeds = [4974, 7737, 6665, 6117, 8559]
    for seed in seeds:
        for i in [0,1]: #loop over baselines: fasttext and glove
            for ibr in ibrs:
                log.append(qsub_launch('midriser',(base_embeds[i], base_embeds_path[i], seed, maker.get_base_outputdir(), rungroup, ibr)))
    log_launch(maker.get_log_name(name, rungroup))

def test0_midriser_10_5_18(name):
    rungroup = 'test0-midriser'
    methods = ['midriser']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    ibrs = [1,2]
    base_embeds = ['fasttext','glove']
    base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
    base_embeds_path = [base_path_ft, base_path_glove]
    seeds = [1]
    for seed in seeds:
        for i in [0,1]: #loop over baselines: fasttext and glove
            for ibr in ibrs:
                log.append(qsub_launch('midriser',(base_embeds[i], base_embeds_path[i], seed, maker.get_base_outputdir(), rungroup, ibr)))
    log_launch(maker.get_log_name(name, rungroup))

def dca_hp_all_params_demo_10_4_18(name):
    rungroup = 'hp_tune_all_demo'
    methods = 'dca'
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    m = [1145, 573, 376]
    k = [2,4,8]
    ibr = 4
    seed = 1234
    tau = [1.0,0.5,2]
    batchsize = [64,32,128]
    gradclip = [0.001,0.01,0.0005,0.002]
    lr = [0.0001,0.001,0.00005,0.00025]  
    base_embeds = ['fasttext','glove']
    base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
    base_embeds_path = [base_path_ft, base_path_glove]
    seeds = [4974, 7737, 6665, 6117, 8559]
    for seed in seeds:
        for i in [0,1]: #loop over baselines: fasttext and glove
            for ibr in ibr:
                log.append(qsub_launch('stochround',(base_embeds[i], base_embeds_path[i], seed, maker.get_base_outputdir(), rungroup, ibr)))
    log_launch(maker.get_log_name(name, rungroup))

def launch_official_stochround_maketime_10_4_18(name):
    rungroup = 'experiment4-1X-seeds'
    methods = ['stochround']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    ibrs = [1,2,4]
    base_embeds = ['fasttext','glove']
    base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
    base_embeds_path = [base_path_ft, base_path_glove]
    seeds = [1944, 7997, 3506]
    for seed in seeds:
        for i in [0,1]: #loop over baselines: fasttext and glove
            for ibr in ibrs:
                log.append(qsub_launch('stochround',(base_embeds[i], base_embeds_path[i], seed, maker.get_base_outputdir(), rungroup, ibr)))
    log_launch(maker.get_log_name(name, rungroup))

def launch_official_stochround_10_3_18(name):
    rungroup = 'experiment2-5X-seeds'
    methods = ['stochround']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    ibrs = [1,2,4]
    base_embeds = ['fasttext','glove']
    base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
    base_embeds_path = [base_path_ft, base_path_glove]
    seeds = [4974, 7737, 6665, 6117, 8559]
    for seed in seeds:
        for i in [0,1]: #loop over baselines: fasttext and glove
            for ibr in ibrs:
                log.append(qsub_launch('stochround',(base_embeds[i], base_embeds_path[i], seed, maker.get_base_outputdir(), rungroup, ibr)))
    log_launch(maker.get_log_name(name, rungroup))

def test0_stochround_10_3_18(name):
    rungroup = 'test0-stochround'
    methods = ['stochround']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    ibrs = [1,2]
    base_embeds = ['fasttext','glove']
    base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
    base_embeds_path = [base_path_ft, base_path_glove]
    seeds = [1]
    for seed in seeds:
        for i in [0,1]: #loop over baselines: fasttext and glove
            for ibr in ibrs:
                log.append(qsub_launch('stochround',(base_embeds[i], base_embeds_path[i], seed, maker.get_base_outputdir(), rungroup, ibr)))
    log_launch(maker.get_log_name(name, rungroup))

def launch_experiment4_2X_seeds_10_1_18(name):
    #date of code Oct 1, 2018
    rungroup = 'experiment4-1X-seeds'
    methods = ['dca','kmeans']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    params['dca'] = dict()
    params['dca']['glove']= [(4,64,0.1),(17,16,0.25),(47,8,0.5),(286,2,1),(286,4,2),(376,8,4)]
    params['dca']['fasttext']= [(4,64,0.1),(13,32,0.25),(47,8,0.5),(286,2,1),(286,4,2),(573,4,4)]
    params['kmeans'] = [(1,10,0.1),(1,4,0.25),(1,2,0.5),(1,1,1),(2,1,2),(4,1,4)]
    for method in methods:
        base_embeds = ['fasttext','glove']
        base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
        base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
        base_embeds_path = [base_path_ft, base_path_glove]
        for i in range(len(base_embeds)):
            seeds = [1944, 7997]
            method_params = params[method][base_embeds[i]] if method == 'dca' else params[method]
            sweep(method, rungroup, [base_embeds[i]], [base_embeds_path[i]], seeds, method_params)
    log_launch(maker.get_log_name(name, rungroup))

def launch_experiment4_1X_seeds_10_1_18(name):
    #date of code Oct 1, 2018
    rungroup = 'experiment4-1X-seeds'
    methods = ['dca','kmeans']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    params['dca'] = dict()
    params['dca']['glove']= [(4,64,0.1),(17,16,0.25),(47,8,0.5),(286,2,1),(286,4,2),(376,8,4)]
    params['dca']['fasttext']= [(4,64,0.1),(13,32,0.25),(47,8,0.5),(286,2,1),(286,4,2),(573,4,4)]
    params['kmeans'] = [(1,10,0.1),(1,4,0.25),(1,2,0.5),(1,1,1),(2,1,2),(4,1,4)]
    for method in methods:
        base_embeds = ['fasttext','glove']
        base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
        base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
        base_embeds_path = [base_path_ft, base_path_glove]
        for i in range(len(base_embeds)):
            seeds = [3506]
            method_params = params[method][base_embeds[i]] if method == 'dca' else params[method]
            sweep(method, rungroup, [base_embeds[i]], [base_embeds_path[i]], seeds, method_params)
    log_launch(maker.get_log_name(name, rungroup))

def test_cfn_2_9_30_18(name):
    #date of code Sept 28, 2018
    #dupe of launch_experiment2_5X_seeds
    rungroup = 'test-cfn-2'
    methods = ['dca','kmeans']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    params['dca'] = dict()
    params['dca']['glove']= [(4,64,0.1)]
    params['dca']['fasttext']= [(4,64,0.1)]
    params['kmeans'] = [(1,10,0.1)]
    for method in methods:
        base_embeds = ['fasttext','glove']
        base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
        base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
        base_embeds_path = [base_path_ft, base_path_glove]
        for i in range(len(base_embeds)):
            seeds = [1]
            method_params = params[method][base_embeds[i]] if method == 'dca' else params[method]
            sweep(method, rungroup, [base_embeds[i]], [base_embeds_path[i]], seeds, method_params)
    log_launch(maker.get_log_name(name, rungroup))


def launch_experiment3_5X_seeds(name):
    #date of code Sept 28, 2018
    #dupe of launch_experiment2_5X_seeds
    rungroup = 'experiment2-5X-seeds'
    methods = ['dca','kmeans']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    params['dca'] = dict()
    params['dca']['glove']= [(4,64,0.1),(17,16,0.25),(47,8,0.5),(286,2,1),(286,4,2),(376,8,4)]
    params['dca']['fasttext']= [(4,64,0.1),(13,32,0.25),(47,8,0.5),(286,2,1),(286,4,2),(573,4,4)]
    params['kmeans'] = [(1,10,0.1),(1,4,0.25),(1,2,0.5),(1,1,1),(2,1,2),(4,1,4)]
    for method in methods:
        base_embeds = ['fasttext','glove']
        base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
        base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
        base_embeds_path = [base_path_ft, base_path_glove]
        for i in range(len(base_embeds)):
            seeds = [1944, 3172, 4258, 7235, 7997]
            method_params = params[method][base_embeds[i]] if method == 'dca' else params[method]
            sweep(method, rungroup, [base_embeds[i]], [base_embeds_path[i]], seeds, method_params)
    log_launch(maker.get_log_name(name, rungroup))

def debug_timing_1_9_28_18(name):
    #date of code Sept 28, 2018
    rungroup = 'debug-timing'
    methods = ['dca']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    params['dca'] = dict()
    params['dca']['glove']= [(4,64,0.1),(17,16,0.25)]
    params['dca']['fasttext']= [(573,4,4)]
    for method in methods:
        base_embeds = ['fasttext','glove']
        base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
        base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
        base_embeds_path = [base_path_ft, base_path_glove]
        for i in range(len(base_embeds)):
            seeds = [100]
            method_params = params[method][base_embeds[i]] if method == 'dca' else params[method]
            sweep(method, rungroup, [base_embeds[i]], [base_embeds_path[i]], seeds, method_params)
    log_launch(maker.get_log_name(name, rungroup))

def relaunch_experiment2_5X_faulty_QA3_9_27_18(name):
    with open('/proj/smallfry/launches/eval/2018-09-26:merged-experiment2-5X-seeds:'+name,'w+') as log_f:
        #94
        log_f.write("qsub -V -b y -wd /proj/smallfry/qsub_logs/eval/2018-09-25:merged-experiment2-5X-seeds:eval-QA-int-simple-synths-official python3.6 /proj/smallfry/git/smallfry/experiments/evaluation/evaluate.py eval-embeddings /proj/smallfry/embeddings/merged-experiment2-5X-seeds/base=glove,method=kmeans,vocab=400000,dim=300,ibr=0.1,bitsperblock=1,blocklen=10,seed=8559,date=2018-09-24,rungroup=experiment2-5X-seeds QA --seed 8559 --epochs 50\n")
        
        #99
        log_f.write("qsub -V -b y -wd /proj/smallfry/qsub_logs/eval/2018-09-25:merged-experiment2-5X-seeds:eval-QA-int-simple-synths-official python3.6 /proj/smallfry/git/smallfry/experiments/evaluation/evaluate.py eval-embeddings /proj/smallfry/embeddings/merged-experiment2-5X-seeds/base=fasttext,method=dca,vocab=400000,dim=300,ibr=2.0,m=286,k=4,seed=8559,date=2018-09-24,rungroup=experiment2-5X-seeds QA --seed 8559 --epochs 50\n")

        #101
        log_f.write("qsub -V -b y -wd /proj/smallfry/qsub_logs/eval/2018-09-25:merged-experiment2-5X-seeds:eval-QA-int-simple-synths-official python3.6 /proj/smallfry/git/smallfry/experiments/evaluation/evaluate.py eval-embeddings /proj/smallfry/embeddings/merged-experiment2-5X-seeds/base=fasttext,method=dca,vocab=400000,dim=300,ibr=0.1,m=4,k=64,seed=8559,date=2018-09-24,rungroup=experiment2-5X-seeds QA --seed 8559 --epochs 50\n")

        #102 
        log_f.write("qsub -V -b y -wd /proj/smallfry/qsub_logs/eval/2018-09-25:merged-experiment2-5X-seeds:eval-QA-int-simple-synths-official python3.6 /proj/smallfry/git/smallfry/experiments/evaluation/evaluate.py eval-embeddings /proj/smallfry/embeddings/merged-experiment2-5X-seeds/base=fasttext,method=kmeans,vocab=400000,dim=300,ibr=2.0,bitsperblock=2,blocklen=1,seed=8559,date=2018-09-24,rungroup=experiment2-5X-seeds QA --seed 8559 --epochs 50\n") 

        #103
        log_f.write("qsub -V -b y -wd /proj/smallfry/qsub_logs/eval/2018-09-25:merged-experiment2-5X-seeds:eval-QA-int-simple-synths-official python3.6 /proj/smallfry/git/smallfry/experiments/evaluation/evaluate.py eval-embeddings /proj/smallfry/embeddings/merged-experiment2-5X-seeds/base=glove,method=kmeans,vocab=400000,dim=300,ibr=0.5,bitsperblock=1,blocklen=2,seed=8559,date=2018-09-24,rungroup=experiment2-5X-seeds QA --seed 8559 --epochs 50\n")

        #104
        log_f.write("qsub -V -b y -wd /proj/smallfry/qsub_logs/eval/2018-09-25:merged-experiment2-5X-seeds:eval-QA-int-simple-synths-official python3.6 /proj/smallfry/git/smallfry/experiments/evaluation/evaluate.py eval-embeddings /proj/smallfry/embeddings/merged-experiment2-5X-seeds/base=glove,method=kmeans,vocab=400000,dim=300,ibr=1.0,bitsperblock=1,blocklen=1,seed=8559,date=2018-09-24,rungroup=experiment2-5X-seeds QA --seed 8559 --epochs 50\n")  

def relaunch_experiment2_5X_faulty_QA2_9_26_18(name):
    '''
    This is a special command creation -- relaunches a faulty job
    '''
    relaunch = []
    relaunch.extend([36,38,55,59,60])
    relaunch.extend(list(range(63,71)))
    with open('/proj/smallfry/launches/eval/2018-09-26:merged-experiment2-5X-seeds:re-eval-faulty-QA','r') as cmd_f:
        with open('/proj/smallfry/launches/eval/2018-09-26:merged-experiment2-5X-seeds:'+name,'w+') as log_f:
            cnt = 1
            line = cmd_f.readline()
            while line:
                if cnt in relaunch:
                    log_f.write(line)
                cnt += 1
                line = cmd_f.readline()

def relaunch_experiment2_5X_faulty_QA_9_26_18(name):
    '''
    This is a special command creation -- relaunches a faulty job
    '''
    relaunch = []
    relaunch.extend([2,14,42,49,52])
    relaunch.extend(list(range(66,131)))
    with open('/proj/smallfry/launches/eval/2018-09-25:merged-experiment2-5X-seeds:eval-QA-int-simple-synths-official','r') as cmd_f:
        with open('/proj/smallfry/launches/eval/2018-09-26:merged-experiment2-5X-seeds:'+name,'w+') as log_f:
            cnt = 1
            line = cmd_f.readline()
            while line:
                if cnt in relaunch:
                    log_f.write(line)
                cnt += 1
                line = cmd_f.readline()
            
def launch_experiment2_5X_baselines_9_25_18(name):
    rungroup = 'experiment2-5X-seeds'
    methods = ['baseline']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    ibr = 32.0
    base_embeds = ['fasttext','glove']
    base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
    base_embeds_path = [base_path_ft, base_path_glove]
    seeds = [4974, 7737, 6665, 6117, 8559]
    for seed in seeds:
        for i in [0,1]: #loop over baselines: fasttext and glove
            log.append(qsub_launch('baseline',(base_embeds[i], base_embeds_path[i], seed, maker.get_base_outputdir(), rungroup, ibr)))
    log_launch(maker.get_log_name(name, rungroup))

def launch_experiment2_5X_seeds(name):
    #date of code Sept 23, 2018
    rungroup = 'experiment2-5X-seeds'
    methods = ['dca','kmeans']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    params['dca'] = dict()
    params['dca']['glove']= [(4,64,0.1),(17,16,0.25),(47,8,0.5),(286,2,1),(286,4,2),(376,8,4)]
    params['dca']['fasttext']= [(4,64,0.1),(13,32,0.25),(47,8,0.5),(286,2,1),(286,4,2),(573,4,4)]
    params['kmeans'] = [(1,10,0.1),(1,4,0.25),(1,2,0.5),(1,1,1),(2,1,2),(4,1,4)]
    for method in methods:
        base_embeds = ['fasttext','glove']
        base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
        base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
        base_embeds_path = [base_path_ft, base_path_glove]
        for i in range(len(base_embeds)):
            seeds = [4974, 7737, 6665, 6117, 8559]
            method_params = params[method][base_embeds[i]] if method == 'dca' else params[method]
            sweep(method, rungroup, [base_embeds[i]], [base_embeds_path[i]], seeds, method_params)
    log_launch(maker.get_log_name(name, rungroup))

def launch_experiment2_5X_seeds_glove(name):
    #date of code Sept 23, 2018
    rungroup = 'experiment2-5X-seeds'
    methods = ['dca','kmeans']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    params['dca'] = [(4,64,0.1),(17,16,0.25),(47,8,0.5),(286,2,1),(286,4,2),(376,4,4)]
    params['kmeans'] = [(1,10,0.1),(1,4,0.25),(1,2,0.5),(1,1,1),(2,1,2),(4,1,4)]
    for method in methods:
        base_embeds = ['glove']
        base_path = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
        base_embeds_path = [base_path]
        seeds = seeds = [4974, 7737, 6665, 6117, 8559]
        method_params = params[method]
        sweep(method, rungroup, base_embeds, base_embeds_path, seeds, method_params)
    log_launch(maker.get_log_name(name, rungroup))

def launch_experiment2_5X_seeds_fasttext(name):
    #date of code Sept 23, 2018
    rungroup = 'experiment2-5X-seeds'
    methods = ['dca','kmeans']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    params['dca'] = [(4,64,0.1),(13,32,0.25),(47,8,0.5),(286,2,1),(286,4,2),(573,4,4)]
    params['kmeans'] = [(1,10,0.1),(1,4,0.25),(1,2,0.5),(1,1,1),(2,1,2),(4,1,4)]
    for method in methods:
        base_embeds = ['fasttext']
        base_path = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
        base_embeds_path = [base_path]
        seeds = seeds = [4974, 7737, 6665, 6117, 8559]
        method_params = params[method]
        sweep(method, rungroup, base_embeds, base_embeds_path, seeds, method_params)
    log_launch(maker.get_log_name(name, rungroup))

def launch_test1_logging(name):
    #date of code Sept 22, 2018
    rungroup = 'test-logging-1'
    methods = ['dca','kmeans']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    params['dca'] = [(3,8,0.1)]
    params['kmeans'] = [(1,1,1)]
    for method in methods:
        base_embeds = ['glove']
        base_path = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000,v=10000'))
        base_embeds_path = [base_path]
        seeds = [int(np.random.random()*1000)]
        method_params = params[method]
        sweep(method, rungroup, base_embeds, base_embeds_path, seeds, method_params, qsub=False)
    log_launch(maker.get_log_name(name, rungroup))


def launch_experiment1_dca_tune_400K(name):
    #date of code Sept 19, 2018
    rungroup = 'experiment1-dca-tune'
    methods = ['dca']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    for method in methods:
        base_embeds = ['fasttext','glove']
        base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
        base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
        base_embeds_path = [base_path_ft, base_path_glove]
        seeds = [int(np.random.random()*10000)]
        sweep(method, rungroup, base_embeds, base_embeds_path, seeds, method_params)
    log_launch(maker.get_log_name(name, rungroup))

def launch_experiment1_dca_tune_missing_pts(name):
    #date of code Sept 20, 2018
    rungroup = 'experiment1-dca-hp-tune'
    methods = ['dca']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = [(119,128),(85,256),(42,256)]
    for method in methods:
        base_embeds = ['fasttext','glove']
        base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
        base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
        base_embeds_path = [base_path_ft, base_path_glove]
        seeds = [3245]
        sweep(method, rungroup, base_embeds, base_embeds_path, seeds, params)
    log_launch(maker.get_log_name(name, rungroup))

def launch_experiment1_kmeans_5X(name):
    #date of code Sept 20, 2018
    rungroup = 'experiment1-5X-seeds'
    methods = ['kmeans']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    params['kmeans'] = [(1,10),(1,4),(1,2),(1,1),(1,2),(1,4)]
    for method in methods:
        base_embeds = ['fasttext','glove']
        base_path_ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
        base_path_glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
        base_embeds_path = [base_path_ft, base_path_glove]
        seeds = [4974, 7737, 6665, 6117, 8559]
        method_params = params[method]
        sweep(method, rungroup, base_embeds, base_embeds_path, seeds, method_params)
    log_launch(maker.get_log_name(name, rungroup))

def launch_debug_githash(name):
    #date of code Sept 18, 2018
    rungroup = 'debug-githash'
    methods = ['dca']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    params['dca'] = [(4,4)]
    for method in methods:
        base_embeds = ['fasttext']
        base_path = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000,v=10000'))
        base_embeds_path = [base_path]
        seeds = [20]
        method_params = params[method]
        sweep(method, rungroup, base_embeds, base_embeds_path, seeds, method_params)
    log_launch(maker.get_log_name(name, rungroup))

def launch_debug_dca_loss(name):
    #date of code Sept 17, 2018
    rungroup = 'debug-dca-loss'
    methods = ['dca']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    params['dca'] = [(16,16),(30,8)]
    for method in methods:
        base_embeds = ['fasttext']
        base_path = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000,v=10000'))
        base_embeds_path = [base_path]
        seeds = [20]
        method_params = params[method]
        sweep(method, rungroup, base_embeds, base_embeds_path, seeds, method_params)
    log_launch(maker.get_log_name(name, rungroup))

def launch2_official_qsub(name):
    #date of code Sept 17, 2018
    rungroup = 'official-test-run-lite-2'
    methods = ['dca','kmeans']
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    params['dca'] = [(16,16),(30,8)]
    params['kmeans'] = [ (1,1),(2,4) ]
    for method in methods:
        base_embeds = ['fasttext']
        glove_path = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
        base_embeds_path = [glove_path]
        seeds = [20]
        method_params = params[method]
        sweep(method, rungroup, base_embeds, base_embeds_path, seeds, method_params)
    log_launch(maker.get_log_name(name, rungroup))

def launch1_official_qsub(name):
    #date of code Sept 17, 2018
    rungroup = 'official-test-run-lite'
    methods = ['dca','kmeans']
    maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    params = dict()
    params['dca'] = [(16,16),(30,8)]
    params['kmeans'] = [ (1,1),(2,4) ]
    for method in methods:
        base_embeds = ['fasttext']
        glove_path = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
        base_embeds_path = [glove_path]
        seeds = [20]
        method_params = params[method]
        sweep(method, rungroup, base_embeds, base_embeds_path, seeds, method_params)
    log_launch(maker.get_log_name(name, rungroup))

def launch1_official(name):
    #date of code Sept 17, 2018
    rungroup = 'official-test-run-lite'
    methods = ['dca','kmeans']
    params = dict()
    params['dca'] = [(16,16),(30,8)]
    params['kmeans'] = [ (1,1),(2,4) ]
    for method in methods:
        base_embeds = ['fasttext']
        glove_path = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
        base_embeds_path = [glove_path]
        seeds = [20]
        method_params = params[method]
        sweep(method, rungroup, base_embeds, base_embeds_path, seeds, method_params, False)
    log_launch(get_log_name(name, rungroup))

def launch1_demo2(name):
    #date of code Sept 16, 2018
    rungroup = 'sweep-6297-test-2'
    methods = ['dca','kmeans']
    params = dict()
    params['dca'] = [(16,16),(30,8)]
    params['kmeans'] = [ (1,1),(2,4) ]
    for method in methods:
        base_embeds = ['fasttext']
        glove_path = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
        base_embeds_path = [glove_path]
        seeds = [6297]
        method_params = params[method]
        sweep(method, rungroup, base_embeds, base_embeds_path, seeds, method_params, False)
    log_launch(get_log_name(name, rungroup))

def launch1_demo(name):
    #date of code Sept 16, 2018
    rungroup = 'sweep-100-test'
    methods = ['dca','kmeans']
    params = dict()
    params['dca'] = [(4,4),(8,8)]
    params['kmeans'] = [ (1,1),(2,4) ]
    for method in methods:
        base_embeds = ['glove']
        glove_path = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000,v=10000'))
        base_embeds_path = [glove_path]
        seeds = [100]
        method_params = params[method]
        sweep(method, rungroup, base_embeds, base_embeds_path, seeds, method_params, False)
    log_launch(get_log_name(name, rungroup))

def launch0_demo_dca(name):
    #date of code Sept 12, 2018
    rungroup = 'demogroup'
    method = 'dca'
    name = name + ':' + maker.get_date_str()+rungroup
    base_embeds = ['glove']
    glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000,v=10000'))
    ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext'))
    base_embeds_path = [glove]
    seeds = [1000]
    mks = [(4,4),(6,4),(6,8)]
    sweep(method, rungroup, base_embeds, base_embeds_path, seeds, mks, False)
    log_launch(name)

def launch0_demo(name):
    #date of code Sept 12, 2018
    rungroup = 'demogroup'
    method = 'kmeans'
    name = name + ':' + maker.get_date_str()+rungroup
    base_embeds = ['glove']
    glove = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000,v=10000'))
    ft = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext'))
    base_embeds_path = [glove]
    seeds = [1000]
    bpb_bl = [(4,1),(2,1),(1,1),(3,6),(1,4),(1,10)]
    sweep(method, rungroup, base_embeds, base_embeds_path, seeds, bpb_bl, False)
    log_launch(name)

def make_kmeans_exp6_10_9_18(name):
    rungroup = 'experiment6-dim-reduc-mini'
    methods = ['kmeans']
    ibrs = [0.1,0.25,0.5,1,2,4,6]
    bpb = [1, 1, 1, 1, 2, 4, 6]
    blkln = [10, 4, 2, 1, 1, 1, 1]
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    configs = []
    for method in methods:
        for i in range(len(ibrs)):
            config = dict()
            config['ibr'] = ibrs[i]
            config['bitsperblock'] = bpb[i]
            config['blocklen'] = blkln[i]
            config['rungroup'] = rungroup
            config['base'] = 'glove'
            config['basepath'] = str(pathlib.PurePath(maker.get_base_embed_path_head(),
             'corpus=text8,method=glove,maxvocab=100000,dim=300,memusage=128,seed=1234,date=2018-10-09,rungroup=experiment6-dim-reduc-mini.txt'))
            config['method'] = method
            config['outputdir'] = maker.get_base_outputdir()
            config['seed'] = 1234
            config['solver'] = 'iterative'
            configs.append(config)
    sweep_configs(configs)
    log_launch(maker.get_log_name(name, rungroup))

def make_optranuni_exp6_10_9_18(name):
    rungroup = 'experiment6-dim-reduc-mini'
    methods = ['optranuni']
    ibrs = [1,2,4,6]
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    configs = []
    for method in methods:
        for i in range(len(ibrs)):
            config = dict()
            config['ibr'] = ibrs[i]
            config['rungroup'] = rungroup
            config['base'] = 'glove'
            config['basepath'] = str(pathlib.PurePath(maker.get_base_embed_path_head(),
             'corpus=text8,method=glove,maxvocab=100000,dim=300,memusage=128,seed=1234,date=2018-10-09,rungroup=experiment6-dim-reduc-mini.txt'))
            config['method'] = method
            config['outputdir'] = maker.get_base_outputdir()
            config['seed'] = 1234
            configs.append(config)
    sweep_configs(configs)
    log_launch(maker.get_log_name(name, rungroup))

def test_optranuni_and_clip_10_11_18(name):
    rungroup = 'test-clipnoquant-and-goldensearch'
    methods = ['optranuni','clipnoquant']
    ibrs = [1,2,4]
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    configs = []
    for method in methods:
        for i in range(len(ibrs)):
            config = dict()
            config['ibr'] = ibrs[i]
            config['rungroup'] = rungroup
            config['base'] = 'glove'
            config['basepath'] = str(pathlib.PurePath(maker.get_base_embed_path_head(),
             'corpus=text8,method=glove,maxvocab=100000,dim=300,memusage=128,seed=1234,date=2018-10-09,rungroup=experiment6-dim-reduc-mini.txt'))
            config['method'] = method
            config['outputdir'] = maker.get_base_outputdir()
            config['seed'] = 1234
            configs.append(config)
    sweep_configs(configs)
    log_launch(maker.get_log_name(name, rungroup))

def test_kmeans_10_11_18(name):
    rungroup = 'test-clipnoquant-and-goldensearch'
    methods = ['kmeans']
    ibrs = [1,2,4]
    bpb = [1,2,4]
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    configs = []
    for method in methods:
        for i in range(len(ibrs)):
            config = dict()
            config['ibr'] = ibrs[i]
            config['rungroup'] = rungroup
            config['base'] = 'glove'
            config['basepath'] = str(pathlib.PurePath(maker.get_base_embed_path_head(),
             'corpus=text8,method=glove,maxvocab=100000,dim=300,memusage=128,seed=1234,date=2018-10-09,rungroup=experiment6-dim-reduc-mini.txt'))
            config['method'] = method
            config['outputdir'] = maker.get_base_outputdir()
            config['seed'] = 1234
            config['bitsperblock'] = bpb[i]
            config['blocklen'] = 1
            configs.append(config)
    sweep_configs(configs)
    log_launch(maker.get_log_name(name, rungroup))

def make_quant_ablation_exp7_10_11_18(name):
    rungroup = 'experiment7-quant-ablation'
    methods = ['optranuni','clipnoquant','kmeans']
    ibrs = [0.5,1,2,4,8]
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    configs = []
    for method in methods:
        for i in range(len(ibrs)):
            config = dict()
            if ibrs[i] < 0.999 and method != 'kmeans':
                continue
            config['ibr'] = ibrs[i]
            config['rungroup'] = rungroup
            config['base'] = 'glove'
            config['basepath'] = str(pathlib.PurePath(maker.get_base_embed_path_head(),
             'corpus=text8,method=glove,maxvocab=100000,dim=300,memusage=128,seed=1234,date=2018-10-09,rungroup=experiment6-dim-reduc-mini.txt'))
            config['method'] = method
            config['outputdir'] = maker.get_base_outputdir()
            config['seed'] = 1234
            config['bitsperblock'] = ibrs[i]
            config['blocklen'] = 1
            if ibrs[i] < 0.999:
                config['bitsperblock'] = 1
                config['blocklen'] = 2
            configs.append(config)
    sweep_configs(configs)
    log_launch(maker.get_log_name(name, rungroup))

def make_baseline_exp7_10_11_18(name):
    rungroup = 'experiment7-quant-ablation'
    methods = ['baseline']
    ibrs = [32]
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    configs = []
    for method in methods:
        for i in range(len(ibrs)):
            config = dict()
            config['ibr'] = ibrs[i]
            config['rungroup'] = rungroup
            config['base'] = 'glove'
            config['basepath'] = str(pathlib.PurePath(maker.get_base_embed_path_head(),
             'corpus=text8,method=glove,maxvocab=100000,dim=300,memusage=128,seed=1234,date=2018-10-09,rungroup=experiment6-dim-reduc-mini.txt'))
            config['method'] = method
            config['outputdir'] = maker.get_base_outputdir()
            config['seed'] = 1234
            configs.append(config)
    sweep_configs(configs)
    log_launch(maker.get_log_name(name, rungroup))


#IMPORTANT!! this line determines which cmd will be run
cmd = [make_baseline_exp7_10_11_18]

parser = argh.ArghParser()
parser.add_commands(cmd)

if __name__ == '__main__':
    parser.dispatch()
