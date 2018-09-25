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

def launch(method, params):
    s = ''
    if method == 'kmeans':
        s = 'python3.6 /proj/smallfry/git/smallfry/experiments/maker/maker.py --method kmeans --base %s --basepath %s --seed %s --outputdir %s --rungroup %s --bitsperblock %s --blocklen %s --ibr %s' % params
    elif method == 'dca':
        s = 'python3.6 /proj/smallfry/git/smallfry/experiments/maker/maker.py --method dca --base %s --basepath %s --seed %s --outputdir %s --rungroup %s --m %s --k %s --ibr %s' % params
    else:
        assert 'bad method name in launch'
    return s

def qsub_launch(method, params):
    return 'qsub -V -b y -wd %s %s ' % (qsub_log_path, launch(method, params))

'''
HELPER METHODS FOR COMMON SWEEP STYLES (and logging)
'''

def log_launch(name):
    log_launch_path = str(pathlib.PurePath( launch_path, name ))
    with open(log_launch_path, 'w+') as llp:
        llp.write('\n'.join(log))

def dca_hyperparam_sweep(bitrates, base_embeds_path, upper_power=8, size_tol=0.15):
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

'''
LAUNCH ROUTINES BELOW THIS LINE =========================
'''

def launch_experiment2_5X_seeds_final_4(name):
    #date of code Sept 24, 2018
    rungroup = 'experiment2-5X-seeds'
    global qsub_log_path
    qsub_log_path = maker.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    methods = 'dca'
    seeds = [6665, 7737, 8559, 8559]
    ibr = [1,2,2,4]
    m = [286,286,286,376]
    k = [2,4,4,8]
    base_embeds = 'fasttext'
    base_path = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'fasttext_k=400000'))
    base_embeds_path = base_path
    for i in range(4):
        log.append(qsub_launch('dca', (base_embeds, base_embeds_path, seeds[i], maker.get_base_outputdir(), rungroup, m[i], k[i], ibr[i])))
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

#IMPORTANT!! this line determines which cmd will be run
cmd = [launch_experiment2_5X_seeds_final_4]

parser = argh.ArghParser()
parser.add_commands(cmd)

if __name__ == '__main__':
    parser.dispatch()
