import argh
import pathlib
import os
import numpy as np
import generate

'''
CORE LAUNCH METHODS: launch and qsub_launch
'''
#globally stores a launch
log = []
#maker cmd creator launch path
launch_path = str(pathlib.PurePath(generate.get_launch_path(), 'gen'))
qsub_log_path = str(pathlib.PurePath(generate.get_qsub_log_path(), 'gen'))

'''
HELPER METHODS FOR COMMON SWEEP STYLES (and logging)
'''

def log_launch(name):
    log_launch_path = str(pathlib.PurePath( launch_path, name ))
    with open(log_launch_path, 'w+') as llp:
        llp.write('\n'.join(log))

def ibr_2_dim(ibr,dim=300):
    compratio = 32/ibr
    return int(np.round(dim/compratio))

def sweep_configs(configs,qsub):
    action_path = str(pathlib.PurePath(os.path.dirname(os.path.realpath(__file__)),'generate.py'))
    for config in configs:
        log.append(generate.launch_config(config,'gen',action_path,qsub))

'''
LAUNCH ROUTINES BELOW THIS LINE =========================
Naming convention: {type}-{date}-{mental-clarity}-{version#}
'''
def generate_exp8_10_14_18(name):
    rungroup = 'exp8-wiki-trained'
    method = 'glove'
    ibrs = [32,1,2,4]
    configs = []
    for ibr in ibrs:
        config = dict()
        config['rungroup'] = rungroup
        config['method'] = method
        config['corpus'] = 'wiki.en.txt'
        config['dim'] = ibr_2_dim(ibr,dim=320)
        config['outputdir'] = generate.get_base_outputdir()
        config['memusage'] = 256
        config['seed'] = 1234
        configs.append(config)
    sweep_configs(configs, False)
    log_launch(generate.get_log_name(name, rungroup))

def generate_exp9_10_15_18(name):
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

def generate_exp9_2_10_16_18(name):
    rungroup = 'exp9-dim-vs-prec'
    method = 'glove'
    dims = [640,20,5]
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

def generate_exp8_v2_10_16_18(name):
    rungroup = 'exp8-wiki-trained'
    method = 'glove'
    ibrs = [32,1,2,4]
    configs = []
    for ibr in ibrs:
        config = dict()
        config['rungroup'] = rungroup
        config['method'] = method
        config['corpus'] = 'wiki.en.txt'
        config['dim'] = ibr_2_dim(ibr,dim=320)
        config['outputdir'] = generate.get_base_outputdir()
        config['memusage'] = 256
        config['seed'] = 1234
        config['lr'] = 0.005
        configs.append(config)
    sweep_configs(configs, False)
    log_launch(generate.get_log_name(name, rungroup))

def generate_exp14_10_22_18(name):
    rungroup = 'exp14-dim-vs-prec-large-scale'
    method = 'glove'
    ibrs = [320,160,80,40,10,640,20,5]
    configs = []
    for ibr in ibrs:
        config = dict()
        config['rungroup'] = rungroup
        config['method'] = method
        config['corpus'] = 'wiki.en.txt'
        config['dim'] = ibr_2_dim(ibr,dim=320)
        config['outputdir'] = generate.get_base_outputdir()
        config['memusage'] = 256
        config['seed'] = 1234
        config['lr'] = 0.005
        configs.append(config)
    sweep_configs(configs, False)
    log_launch(generate.get_log_name(name, rungroup))

#IMPORTANT!! this line determines which cmd will be run
cmd = [generate_exp14_10_22_18]

parser = argh.ArghParser()
parser.add_commands(cmd)

if __name__ == '__main__':
    parser.dispatch()
