import argh
import pathlib
import glob
import os
import evaluate

'''
CORE LAUNCH METHODS: launch and qsub_launch
'''
#globally stores a launch
log = []
#evaluate cmd creator launch path
launch_path = str(pathlib.PurePath(evaluate.get_launch_path(), 'eval'))
qsub_log_path = str(pathlib.PurePath(evaluate.get_qsub_log_path(), 'eval'))

'''
HELPER METHODS FOR COMMON SWEEP STYLES (and logging)
'''
def log_launch(name,batchsize=1):
    if batchsize == 1:
        log_launch_path = str(pathlib.PurePath( launch_path, name ))
        with open(log_launch_path, 'w+') as llp:
            llp.write('\n'.join(log))
    elif batchsize > 1:
        assert not 'qsub' in log[0],"Do NOT use batching with qsub"
        log_launch_path = str(pathlib.PurePath( launch_path, name ))
        for i in range(len(log)):
            if i % (batchsize+1) == batchsize:
                log.insert(i,"wait")
            else:
                log[i] = f"{log[i]} &"
    else:
        raise ValueError(f"batch size {batchsize} in cmd creator must be pos. int.")
    with open(log_launch_path, 'w+') as llp:
        llp.write('\n'.join(log))
        
def sweep_configs(configs,qsub):
    action_path = str(pathlib.PurePath(os.path.dirname(os.path.realpath(__file__)),'evaluate.py'))
    for config in configs:
        log.append(evaluate.launch_config(config,'eval',action_path,qsub))

'''
LAUNCH ROUTINES BELOW THIS LINE =========================
'''
def launch_eval_exp11_10_17_18(name):
    rungroup = '2018-10-17-exp11-stoch-benchmarks'
    evaltypes = ['intrinsics']
    embs = evaluate.get_all_embs_in_rg(rungroup)
    configs = []
    for evaltype in evaltypes:
        for emb in embs:
            config = dict()
            config['evaltype'] = evaltype
            config['embedpath'] = emb
            config['seed'] = 1234
            configs.append(config)
    sweep_configs(configs, False)
    log_launch(evaluate.get_log_name(name,rungroup),batchsize=3)

def launch_eval_exp9_10_16_18(name):
    rungroup = '2018-10-16-exp9-dim-vs-prec-quantized'
    evaltypes = ['intrinsics']
    embs = evaluate.get_all_embs_in_rg(rungroup)
    configs = []
    for evaltype in evaltypes:
        for emb in embs:
            config = dict()
            config['evaltype'] = evaltype
            config['embedpath'] = emb
            config['seed'] = 1234
            configs.append(config)
    sweep_configs(configs, False)
    log_launch(evaluate.get_log_name(name,rungroup),batchsize=3)

def launch_eval_exp8_10_17_18(name):
    rungroup = 'merged-exp8-wiki-trained'
    evaltypes = ['QA','intrinsics']
    embs = evaluate.get_all_embs_in_rg(rungroup)
    configs = []
    for evaltype in evaltypes:
        for emb in embs:
            config = dict()
            config['evaltype'] = evaltype
            config['embedpath'] = emb
            config['seed'] = 1234
            configs.append(config)
    sweep_configs(configs, False)
    log_launch(evaluate.get_log_name(name,rungroup),batchsize=3)

def launch_eval_test1_10_17_18(name):
    rungroup = '2018-10-17-test-1-DBG'
    evaltypes = ['QA','intrinsics']
    embs = evaluate.get_all_embs_in_rg(rungroup)
    configs = []
    for evaltype in evaltypes:
        for emb in embs:
            config = dict()
            config['evaltype'] = evaltype
            config['embedpath'] = emb
            config['seed'] = 1234
            config['epochs'] = 1
            configs.append(config)
    sweep_configs(configs, False)
    log_launch(evaluate.get_log_name(name,rungroup),batchsize=3)

def launch_eval_synth_exp11_10_17_18(name):
    rungroup = '2018-10-17-exp11-stoch-benchmarks'
    evaltypes = ['synthetics']
    embs = evaluate.get_all_embs_in_rg(rungroup)
    configs = []
    for evaltype in evaltypes:
        for emb in embs:
            config = dict()
            config['evaltype'] = evaltype
            config['embedpath'] = emb
            config['seed'] = 1234
            configs.append(config)
    sweep_configs(configs, False)
    log_launch(evaluate.get_log_name(name,rungroup),batchsize=3)

    def launch_eval_exp8_naive_10_19_18(name):
    rungroup = 'merged-exp8-wiki-trained'
    evaltypes = ['QA','intrinsics']
    embs = evaluate.get_all_embs_in_rg(rungroup)
    configs = []
    for evaltype in evaltypes:
        for emb in embs:
            if not 'naive' in emb:
                continue
            config = dict()
            config['evaltype'] = evaltype
            config['embedpath'] = emb
            config['seed'] = 1234
            configs.append(config)
    sweep_configs(configs, False)
    log_launch(evaluate.get_log_name(name,rungroup),batchsize=3)

#IMPORTANT!! this line determines which cmd will be run
cmd = [launch_eval_exp8_naive_10_19_18]

parser = argh.ArghParser()
parser.add_commands(cmd)

if __name__ == '__main__':
    parser.dispatch()
