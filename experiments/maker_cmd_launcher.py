import os
import argh
import pathlib
import maker

'''
CORE LAUNCH METHODS: launch and qsub_launch
'''
log = []

def launch(method, params):
    s = ''
    if method == 'kmeans':
        s = 'python3.6 /proj/smallfry/git/smallfry/experiments/maker.py --method kmeans --base %s --basepath %s --seed %s --outputdir %s --rungroup %s --bitsperblock %s --blocklen %s' % params
    if method == 'dca':
        s = 'python3.6 /proj/smallfry/git/smallfry/experiments/maker.py --method dca --base %s --basepath %s --seed %s --outputdir %s --rungroup %s --m %s --k %s' % params
    else:
        assert 'bad method name in launch'
    log.append(s)
    os.system(s)
    return s

def qsub_launch(method, params):
    return 'qsub '+launch(method, params)

'''
GLOBAL PATHS CODED HERE
'''
base_embed_path_head = '/proj/smallfry/base_embeddings'
launch_path = '/proj/smallfry/launches/'
base_outputdir = '/proj/smallfry/embeddings'

'''
HELPER METHODS FOR COMMON SWEEP STYLES (and logging)
'''
def log_launch(name):
    log_launch_path = str(pathlib.PurePath( launch_path, name ))
    with open(log_launch_path, 'w+') as llp:
        llp.write('\n'.join(log))

def sweep(method, rungroup, base_embeds, base_embeds_path, seeds, params, qsub=True):
    l = qsub_launch if qsub else launch
    for seed in seeds:
        for e in range(len(base_embeds)):
            for p in params:
                l(method,(
                        base_embeds[e],
                        base_embeds_path[e],
                        seed,
                        base_outputdir,
                        rungroup,
                        p[0],
                        p[1]))

'''
LAUNCH ROUTINES BELOW THIS LINE =========================
'''


def launch0_demo_dca(name):
    #date of code Sept 12, 2018
    rungroup = 'demogroup'
    method = 'dca'
    name = name + ':' + maker.get_date_str()+rungroup
    base_embeds = ['glove']
    glove = str(pathlib.PurePath(base_embed_path_head, 'glove_k=400000,v=10000'))
    ft = str(pathlib.PurePath(base_embed_path_head, 'fasttext'))
    base_embeds_path = [glove]
    seeds = [1000]
    mks = [(4,4),(6,4),(6,8)]
    sweep(method, rungroup, base_embeds, base_embeds_path, seeds, mks, False)
    log_launch(name)



def launch0_demo(name):
    #date of code Sept 12, 2018
    rungroup = 'demogroup'
    method = 'kmeans'
    name = name + ':' + get_date_str()+rungroup
    base_embeds = ['glove']
    glove = str(pathlib.PurePath(base_embed_path_head, 'glove_k=400000,v=10000'))
    ft = str(pathlib.PurePath(base_embed_path_head, 'fasttext'))
    base_embeds_path = [glove]
    seeds = [1000]
    bpb_bl = [(4,1),(2,1),(1,1),(3,6),(1,4),(1,10)]
    sweep(method, rungroup, base_embeds, base_embeds_path, seeds, bpb_bl, False)
    log_launch(name)


#IMPORTANT!! this line determines which cmd will be run
cmd = [launch0_demo_dca]

parser = argh.ArghParser()
parser.add_commands(cmd)

if __name__ == '__main__':
    parser.dispatch()
