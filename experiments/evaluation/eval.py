'''
GENERAL PURPOSE EMBEDDINGS EVALUATION 
'''
from smallfry.utils import load_embeddings
import re
import uuid
import json
import sys
import datetime
import time
import pathlib
import os
import subprocess
from subprocess import check_output


#TODO: Ponder this, should we overwrite evals that already exist, or error out? I like err out


def eval(embed_path, evaltype, eval_log_path, eval_params=None):
    #NOTE: embed_path refers to TOP-LEVEL embedding directory path, NOT the path to the .txt
    if evaltype == 'QA':
        qa_results = eval_qa(embed_path, fetch_dim(embed_path), eval_params['seed'])
        embed_name = os.path.basename(embed_path)
        qa_results_path = str(pathlib.PurePath(embed_path, embed_name+'_results-qa.json'))
        with open(qa_results_path, 'w+') as qa_results_f:
            qa_results_f.write(json.dumps(qa_results))

    elif evaltype == 'intrinsics':
        pass
    elif evaltype == 'synthetics':
        pass
    else:
        assert 'bad evaltype given to eval()'


### HELPERS BELOW ###

def fetch_dim(embed_path):
    embed_name = os.path.basename(embed_path)
    maker_config_path = str(pathlib.PurePath(embed_path, embed_name+'_config.json'))
    maker_config = dict()
    with open(maker_config_path, 'r') as maker_config_f:
        maker_config = json.loads(maker_config_f.read())
    return maker_config['dim']

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


#### SPECIFC EVAL TYPES BELOW ###

def eval_qa(word_vectors_path, dim, seed, finetune_top_k=0, extra_args=""):

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

def eval_intrinsics():
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
    all_tasks = similarity_tasks + analogy_tasks

    # This line below is a bit jenky since it assumes `testsets` relative to this file.
    # Should be fine since that data is pulled with the repo.
    path_to_tasks = [os.path.dirname(os.path.abspath(__file__)) + "/testsets/" + x for x in tasks_to_use]

    results = ""
    results_dict = {}
    for task_path in path_to_tasks:
        if os.path.basename(task_path) in analogy_tasks:
            output = evaluate_analogy(word_vectors, task_path)
        elif os.path.basename(task_path) in similarity_tasks:
            output = evaluate_similarity(word_vectors, task_path)
        else:
           lpmf_print("%s not in list of similarity or analogy tasks." % os.path.basename(task_path))
        partial_result = "%s - %s\n" % (os.path.basename(task_path), str(output))
        results += partial_result
        lpmf_print(partial_result)

        task_name = os.path.basename(task_path).replace(".txt", "")
        task_name = task_name.replace("_", "-")
        if type(output) == list or type(output) == tuple:
            results_dict[task_name + "-add"] = output[0]
            results_dict[task_name + "-mul"] = output[1]
        else:
            results_dict[task_name] = output
            
        lpmf_print("Results:")
        lpmf_print("------------------------------")        
        lpmf_print(results)
        lpmf_print("------------------------------")
        return results_dict



def eval_synthetics():
    pass