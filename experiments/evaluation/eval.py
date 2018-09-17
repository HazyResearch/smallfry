'''
GENERAL PURPOSE EMBEDDINGS EVALUATION SCRIPT
'''
import re
import uuid
import json
import sys
import datetime
import time
import pathlib
import os
import subprocess
import argh
import numpy as np
from subprocess import check_output
from hyperwords import ws_eval, analogy_eval
from hyperwords.representations.embedding import *
from smallfry.utils import load_embeddings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import experimental_utils



#TODO: Ponder this, should we overwrite evals that already exist, or error out? I like err out
#TODO: Change from argh to argparse?

def eval_embeddings(embed_path, evaltype, eval_log_path, eval_params=None):
    '''
    This is the front-end routine for experimental evaluation. 
    For each acceptable experiment type, denoted with 'evaltype', it dispatches
    the appropriate evaluation subroutine.
    Finally, results are written to file.
    As of Sept. 16, valid 'evaltype' selections are: 'QA' OR 'intrinsics' OR 'synthetics'
    NOTE: 'embed_path' refers to the TOP-LEVEL embedding directory path, NOT the path to the .txt embeddings file
    '''
    results = None
    #NOTE: embed_path refers to TOP-LEVEL embedding directory path, NOT the path to the .txt
    if evaltype == 'QA':
        results = eval_qa(embed_path, fetch_dim(embed_path), eval_params['seed'])

    elif evaltype == 'intrinsics':
        results = eval_intrinsics(embed_path)

    elif evaltype == 'synthetics':
        results = eval_synthetics(embed_path)
    else:
        assert 'bad evaltype given to eval()'

    results_to_file(embed_path, evaltype, results)


'''
HELPERS BELOW
'''

def results_to_file(embed_path, results_type, results):
    embed_name = os.path.basename(embed_path)
    results_file = '%s_results-%s.json' % (embed_name, results_type)
    results_path = str(pathlib.PurePath(embed_path, results_file))
    with open(results_path, 'w+') as results_f:
            results_f.write(json.dumps(results)) 

def fetch_embeds_4_eval(embed_path):
    embed_name = os.path.basename(embed_path)
    embed_txt_path = str(pathlib.PurePath(embed_path, embed_name+'.txt'))
    embeds, wordlist = load_embeddings(embed_txt_path) #clarify what load_embeddings returns
    assert len(embeds) == len(wordlist), 'Embeddings and wordlist have different lengths in eval.py'
    return embeds, wordlist

def fetch_dim(embed_path):
    embed_name = os.path.basename(embed_path)
    maker_config_path = str(pathlib.PurePath(embed_path, embed_name+'_config.json'))
    maker_config = dict()
    with open(maker_config_path, 'r') as maker_config_f:
        maker_config = json.loads(maker_config_f.read())
    return maker_config['dim']

def fetch_base_embed_path(embed_path):
    embed_name = os.path.basename(embed_path)
    maker_config_path = str(pathlib.PurePath(embed_path, embed_name+'_config.json'))
    maker_config = dict()
    with open(maker_config_path, 'r') as maker_config_f:
        maker_config = json.loads(maker_config_f.read())
    return maker_config['basepath']

def eval_print(message):
    callername = sys._getframe().f_back.f_code.co_name
    tsstring = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print("%s-%s : %s" % (tsstring, callername, message))
    sys.stdout.flush()

def perform_command_local(command):
    out = check_output(command, stderr=subprocess.STDOUT, shell=True).decode("utf-8") 
    return out


def get_drqa_directory():
        return "/proj/smallfry/embeddings_benchmark/DrQA/"

def get_relation_directory():
    return "/proj/smallfry/embeddings_benchmark/tacred-relation/"

def get_sentiment_directory():
    return "/proj/smallfry/embeddings_benchmark/compositional_code_learning/"


# Evaluate similarity -- ROUTINE WRITTEN BY MAXLAM
# -----------------------------------------
# word_vectors - dictionary where keys are words, values are word vectors.
# task_path - path to similarity dataset
# return - similarity score
def evaluate_similarity(word_vectors, task_path):
    print("Evaluating similarity: %s" % task_path)
    assert os.path.exists(task_path)
    data = ws_eval.read_test_set(task_path)
    representation = BootstrapEmbeddings(word_vectors)
    return ws_eval.evaluate(representation, data)

# Evaluate analogy -- ROUTINE WRITTEN BY MAXLAM
# -----------------------------------------
# word_vectors - dictionary where keys are words, values are word vectors.
# task_path - path to similarity dataset
# return - similarity score
def evaluate_analogy(word_vectors, task_path):
    print("Evaluating analogy: %s" % task_path)
    assert os.path.exists(task_path)
    data = analogy_eval.read_test_set(task_path)
    xi, ix = analogy_eval.get_vocab(data)        
    representation = BootstrapEmbeddings(word_vectors)
    return analogy_eval.evaluate(representation, data, xi, ix)


'''
CORE EVALUATION ROUTINES
a new routine must be added for each evaltype!
'''

def eval_qa(word_vectors_path, dim, seed, finetune_top_k=0, extra_args=""):
    '''Calls DrQA's training routine'''

    def to_dict(text):
        result = {}    
        f1_scores = []
        ems = []
        for line in text.splitlines():
            matches = re.findall("F1 = ([0-9]+.[0-9]+)", line)
            if len(matches) != 0:
                f1_scores.append(float(matches[0]))
        for line in text.splitlines():
            matches = re.findall("EM = ([0-9]+.[0-9]+)", line)
            if len(matches) != 0:
                ems.append(float(matches[0]))

        result["all-f1s"] = f1_scores
        result["all-ems"] = ems
        result["max-em"] = max(ems)
        result["max-f1"] = max(f1_scores)
        return json.dumps(result)

    # Evaluate on the word vectors
    cd_dir = "cd %s" % get_drqa_directory()

    # Write intermediate training results to temporary output file
    unique_temp_output_filename = str(uuid.uuid4())
    intermediate_output_file_path = "/%s/%s.txt" % ("tmp", unique_temp_output_filename)
    eval_print("Writing intermediate training to output path: %s" % intermediate_output_file_path)
    
    # WARNING: REALLY DANGEROUS SINCE MAKES ASSUMPTIONS ABOUT 
    # FILEPATHS AND THEIR EXISTENCE
    python_command = "CUDA_HOME=/usr/local/cuda-8.0 python3.6 scripts/reader/train.py --random-seed %d --embedding-dim %d  --embed-dir=  --embedding-file %s  --num-epochs 50 --tune-partial %d %s 2>&1 | tee %s" % (seed, dim, word_vectors_path, finetune_top_k, extra_args, intermediate_output_file_path)
    full_command = " && ".join([cd_dir, python_command])
    eval_print("Executing: %s" % full_command)
    text = perform_command_local(full_command)
    eval_print("==============================")
    eval_print("Output of DrQA run:")
    eval_print("==============================")
    eval_print(text)
    eval_print("==============================")
    eval_print("End output of DrQA run")
    eval_print("==============================")

    rtn_dict = to_dict(text)
    rtn_dict['full_log'] = text

    return rtn_dict

def eval_intrinsics(embed_path):
    '''Evaluates intrinsics benchmarks given embed_path'''

    #load embeddings and make into dict for intrinsic routines
    embeds, wordlist = fetch_embeds_4_eval(embed_path)
    word_vectors = { wordlist[i] : embeds[i] for i in range(len(embeds)) }

    # Get intrinsic tasks to evaluate on        
    similarity_tasks = [
            "bruni_men.txt",
            "luong_rare.txt",
            "radinsky_mturk.txt",
            "simlex999.txt",
            "ws353_relatedness.txt",
            "ws353_similarity.txt"
    ]
    analogy_tasks = [
            "google_caseinsens.txt",
            "msr.txt"
    ]

    # This line below is a bit jenky since it assumes `testsets` relative to this file.
    # Should be fine since that data is pulled with the repo.
    all_tasks = analogy_tasks + similarity_tasks
    path_to_tasks = [os.path.dirname(os.path.abspath(__file__)) + "/testsets/" + x for x in all_tasks  ]

    results = ""
    results_dict = {}
    for task_path in path_to_tasks:
        if os.path.basename(task_path) in analogy_tasks:
            output = evaluate_analogy(word_vectors, task_path)
        elif os.path.basename(task_path) in similarity_tasks:
            output = evaluate_similarity(word_vectors, task_path)
        else:
            eval_print("%s not in list of similarity or analogy tasks." % os.path.basename(task_path))
        partial_result = "%s - %s\n" % (os.path.basename(task_path), str(output))
        results += partial_result
        eval_print(partial_result)

        task_name = os.path.basename(task_path).replace(".txt", "")
        task_name = task_name.replace("_", "-")
        if type(output) == list or type(output) == tuple:
            results_dict[task_name + "-add"] = output[0]
            results_dict[task_name + "-mul"] = output[1]
        else:
            results_dict[task_name] = output
            
    eval_print("Results:")
    eval_print("------------------------------")        
    eval_print(results)
    eval_print("------------------------------")
    return results_dict

def eval_synthetics(embed_path):
    '''Evaluates synthetics'''
    embeds, wordlist = fetch_embeds_4_eval(embed_path)
    base_embeds, base_wordlist = fetch_embeds_4_eval(fetch_base_embed_path(embed_path))

    res_rtn = dict()
    res_rtn['embed-fro'] = np.linalg.norm(base_embeds-embeds)
    res_rtn['mean'] = np.mean(embeds)
    res_rtn['var'] = np.var(embeds)
    return res_rtn


parser = argh.ArghParser()
parser.add_commands([eval_embeddings])

if __name__ == '__main__':
    parser.dispatch()
