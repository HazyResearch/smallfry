import argh
import pathlib
import os
import numpy as np
import maker

'''
CORE LAUNCH VARIABLES: logging a launch
'''
#globally stores a launch
log = []
#maker cmd creator launch path
launch_path = str(pathlib.PurePath(maker.get_launch_path(), 'maker'))

'''
HELPER METHODS FOR COMMON SWEEP STYLES (and logging)
'''

def log_launch(name,batchsize=1):
    if batchsize == 1:
        log_launch_path = str(pathlib.PurePath( launch_path, name ))
        with open(log_launch_path, 'w+') as llp:
            llp.write('\n'.join(log))
    elif batchsize > 1:
        log_launch_path = str(pathlib.PurePath( launch_path, name ))
        for i in range(len(log)):
            if i % 3 == 2:
                log.insert(i,"wait")
            else:
                log[i] = f"{log[i]} &"
    else:
        raise ValueError(f"batch size {batchsize} in cmd creator must be pos. int.")
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

def sweep_configs(configs,qsub):
    action_path = str(pathlib.PurePath(os.path.dirname(os.path.realpath(__file__)),'maker.py'))
    for config in configs:
        log.append(maker.launch_config(config,'maker',action_path,qsub))

'''
LAUNCH ROUTINES BELOW THIS LINE =========================
Naming convention: {type}-{date}-{mental-clarity}-{version#}
'''

def make_exp9_10_15_18(name):
    rungroup = 'exp9-dim-vs-prec-quantized'
    method = 'optranuni'
    embs = maker.get_all_embs_in_rg('2018-10-15-exp9-dim-vs-prec')
    configs = []
    for emb in embs:
        maker_config = maker.fetch_maker_config(emb)
        prec = 320/maker_config['dim']
        config = dict()
        config['base'] = 'glove'
        config['basepath'] = maker.fetch_embeds_txt_path(emb)
        config['rungroup'] = rungroup
        config['method'] = method
        config['ibr'] = prec
        config['outputdir'] = maker.get_base_outputdir()
        config['seed'] = 1234
        configs.append(config)
    sweep_configs(configs, False)
    log_launch(maker.get_log_name(name, rungroup))

def make_exp5_10_15_18(name):
    rungroup = 'experiment5-dca-hp-tune'
    method = 'dca'
    base = 'glove'
    basepath = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
    seed = 1234
    outputdir = maker.get_base_outputdir()
    configs = []
    m = 573
    k = 4
    ibr = 4.0
    lrs = [1e-1,1e-2,1e-6,1e-7]
    for lr in lrs:
        config = dict()
        config['base'] = base
        config['method'] = method
        config['basepath'] = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
        config['rungroup'] = rungroup
        config['method'] = method
        config['ibr'] = ibr
        config['outputdir'] = maker.get_base_outputdir()
        config['seed'] = seed
        config['m'] = m
        config['k'] = k
        config['lr'] = lr
        configs.append(config)

    lrs = [1e-3,1e-4,1e-5]
    batchsizes = [64,32]
    gradclips = [0.01,0.1,1000]
    taus = [0.5,0.25]
    for lr in lrs:
        for bs in batchsizes:
            for gradclip in gradclips:
                for tau in taus:
                    config = dict()
                    config['base'] = base
                    config['method'] = method
                    config['basepath'] = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000'))
                    config['rungroup'] = rungroup
                    config['method'] = method
                    config['ibr'] = ibr
                    config['outputdir'] = maker.get_base_outputdir()
                    config['seed'] = seed
                    config['m'] = m
                    config['k'] = k
                    config['lr'] = lr
                    config['batchsize'] = bs
                    config['gradclip'] = gradclip
                    config['tau'] = tau
                    configs.append(config)

    sweep_configs(configs, False)
    log_launch(maker.get_log_name(name, rungroup))

    def make_exp11_10_16_18(name):
    rungroup = 'exp11-stoch-benchmarks'
    methods = ['stochoptranuni','optranuni','kmeans','clipnoquant']
    brs = [1,2,4]
    emb = str(pathlib.PurePath(maker.get_base_embed_path_head(), 'glove_k=400000')))
    configs = []
    for br in brs:
        for method in methods:
            config = dict()
            config['base'] = 'glove'
            config['basepath'] = emb
            config['rungroup'] = rungroup
            config['method'] = method
            config['ibr'] = br
            config['blocklen'] = 1
            config['bitsperblock'] = br
            config['outputdir'] = maker.get_base_outputdir()
            config['seed'] = 1234
            configs.append(config)
    #add baseline
    config = dict()
    config['base'] = 'glove'
    config['basepath'] = embs[0]
    config['rungroup'] = rungroup
    config['method'] = 'baseline'
    config['ibr'] = 32
    config['outputdir'] = maker.get_base_outputdir()
    config['seed'] = 1234

    sweep_configs(configs, False)
    log_launch(maker.get_log_name(name, rungroup))

#IMPORTANT!! this line determines which cmd will be run
cmd = [make_exp5_10_15_18]

parser = argh.ArghParser()
parser.add_commands(cmd)

if __name__ == '__main__':
    parser.dispatch()
