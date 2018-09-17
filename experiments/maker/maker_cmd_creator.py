import argh
import pathlib
import maker

'''
CORE LAUNCH METHODS: launch and qsub_launch
'''
#globally stores a launch
log = []
#maker cmd creator launch path
launch_path = str(pathlib.PurePath(maker.get_launch_path(), 'maker'))

def launch(method, params):
    s = ''
    if method == 'kmeans':
        s = 'python3.6 /proj/smallfry/git/smallfry/experiments/maker/maker.py --method kmeans --base %s --basepath %s --seed %s --outputdir %s --rungroup %s --bitsperblock %s --blocklen %s' % params
    elif method == 'dca':
        s = 'python3.6 /proj/smallfry/git/smallfry/experiments/maker/maker.py --method dca --base %s --basepath %s --seed %s --outputdir %s --rungroup %s --m %s --k %s' % params
    else:
        assert 'bad method name in launch'
    return s

def qsub_launch(method, params):
    return 'qsub -V -b y -wd /proj/smallfry/qsub_logs '+launch(method, params)


'''
HELPER METHODS FOR COMMON SWEEP STYLES (and logging)
'''
def log_launch(name):
    log_launch_path = str(pathlib.PurePath( launch_path, name ))
    with open(log_launch_path, 'w+') as llp:
        llp.write('\n'.join(log))

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
                        p[1]))
                log.append(cmd)

def get_log_name(name, rungroup):
    return maker.get_date_str() + ':' + rungroup + ':' + name

'''
LAUNCH ROUTINES BELOW THIS LINE =========================
'''

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
cmd = [launch1_official]

parser = argh.ArghParser()
parser.add_commands(cmd)

if __name__ == '__main__':
    parser.dispatch()
