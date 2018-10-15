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
qsub_preamble = "qsub -V -b y -wd"

def launch_config(config, qsub=False):
    s = ''
    global qsub_preamble
    maker_path = str(pathlib.PurePath(os.path.dirname(os.path.realpath(__file__)),'generate.py'))
    python_maker_cmd = 'python %s' % maker_path
    flags = [python_maker_cmd]
    for key in config.keys():
        flags.append(f"--{key} {config[key]}")
    s = " ".join(flags)
    s = f"{qsub_preamble} {qsub_log_path} {s}" if qsub else s
    return s

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
'''

def generate_dim_reduction_exp6_10_9_18(name):
    rungroup = 'experiment6-dim-reduc-mini'
    method = 'glove'
    ibrs = [0.1,0.25,0.5,1,2,4]
    global qsub_log_path
    qsub_log_path = generate.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    configs = []
    for ibr in ibrs:
        config = dict()
        config['rungroup'] = rungroup
        config['method'] = method
        config['corpus'] = 'text8'
        config['dim'] = ibr_2_dim(ibr)
        config['outputdir'] = generate.get_base_outputdir()
        config['seed'] = 1234
        configs.append(config)
    sweep_configs(configs)
    log_launch(generate.get_log_name(name, rungroup))

def generate_dim_reduction2_exp6_10_9_18(name):
    rungroup = 'experiment6-dim-reduc-mini'
    method = 'glove'
    ibrs = [6, 32]
    global qsub_log_path
    qsub_log_path = generate.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    configs = []
    for ibr in ibrs:
        config = dict()
        config['rungroup'] = rungroup
        config['method'] = method
        config['corpus'] = 'text8'
        config['dim'] = ibr_2_dim(ibr)
        config['outputdir'] = generate.get_base_outputdir()
        config['seed'] = 1234
        configs.append(config)
    sweep_configs(configs)
    log_launch(generate.get_log_name(name, rungroup))

def generate_dim_reduc_exp7_quant_ablation_10_11_18(name):
    rungroup = 'experiment7-quant-ablation'
    method = 'glove'
    ibrs = [0.5,1,2,4,8]
    global qsub_log_path
    qsub_log_path = generate.prep_qsub_log_dir(qsub_log_path, name, rungroup)
    configs = []
    for ibr in ibrs:
        config = dict()
        config['rungroup'] = rungroup
        config['method'] = method
        config['corpus'] = 'text8'
        config['dim'] = ibr_2_dim(ibr)
        config['outputdir'] = generate.get_base_outputdir()
        config['seed'] = 1234
        configs.append(config)
    sweep_configs(configs)
    log_launch(generate.get_log_name(name, rungroup))

def generate_test_10_13_18(name):
    rungroup = 'test-refactor'
    method = 'glove'
    ibrs = [2,32,64]
    configs = []
    for ibr in ibrs:
        config = dict()
        config['rungroup'] = rungroup
        config['method'] = method
        config['corpus'] = 'text8'
        config['dim'] = ibr_2_dim(ibr)
        config['outputdir'] = generate.get_base_outputdir()
        config['seed'] = 1234
        configs.append(config)
    sweep_configs(configs, False)
    log_launch(generate.get_log_name(name, rungroup))



#IMPORTANT!! this line determines which cmd will be run
cmd = [generate_test_10_13_18]

parser = argh.ArghParser()
parser.add_commands(cmd)

if __name__ == '__main__':
    parser.dispatch()
