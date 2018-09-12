import os
import argh

log = []

def launch(method, params):
    s = ''
    if method == 'kmeans':
        s = 'python /proj/smallfry/git/smallfry/experiments/maker.py --method kmeans --base %s --basepath %s --seed %s --outputdir %s --rungroup %s --bitsperblock %s --blocklen %s' % params
    if method == 'dca':
        s = 'python /proj/smallfry/git/smallfry/experiments/maker.py --method kmeans --base %s --basepath %s --seed %s --outputdir %s --rungroup %s --m %s --k %s' % params
    else:
        assert 'bad method name in launch'
    log.append(s)
    return s

def qsub_launch(method, params):
    return 'qsub '+launch(method, params)

base_embed_path_head = '/proj/smallfry/base_embeddings'
launch_path = '/proj/smallfry/launches/'

def log_launch(name):
    log_launch_path = str(pathlib.PurePath( launch_path, name ))
    with open(log_launch_path, 'w+') as llp:
        llp.write('\n'.join(log))

def kmeans_sweep():

def dca_sweep():



'''
LAUNCHES BELOW THIS LINE
'''

def launch0_demo(name):
    rungroup = 'demogroup'
    method = ['kmeans']
    glove = [str(pathlib.PurePath(base_embed_path_head, 'glove'))
    ft = [str(pathlib.PurePath(base_embed_path_head, 'fasttext'))
    base_embeds = [glove, ft]
    seeds = [1000]
    bpb_bl = [(4,1),(2,1),(1,1),(3,6),(1,4),(1,10)]


#IMPORTANT this line determines which cmd will be run
cmd = [launch0_demo]

parser = argh.ArghParser()
parser.add_commands(cmd)

if __name__ == '__main__':
    parser.dispatch()
