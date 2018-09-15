'''
GENERAL PURPOSE EMBEDDINGS EVALUATION 
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
from subprocess import check_output
from hyperwords import ws_eval, analogy_eval
from hyperwords.representations.embedding import *
from smallfry.utils import load_embeddings



#TODO: Ponder this, should we overwrite evals that already exist, or error out? I like err out


def eval_embeddings(embed_path, evaltype, eval_log_path, eval_params=None):
    results = None
    #NOTE: embed_path refers to TOP-LEVEL embedding directory path, NOT the path to the .txt
    if evaltype == 'QA':
        results = eval_qa(embed_path, fetch_dim(embed_path), eval_params['seed'])

    elif evaltype == 'intrinsics':
        print("HELLO")
        embed_name = os.path.basename(embed_path)
        print("name: "+embed_name)
        embed_txt_path = str(pathlib.PurePath(embed_path, embed_name+'.txt'))
        print(embed_txt_path)
        embeds, wordlist = load_embeddings(embed_txt_path)
        assert len(embeds) == len(wordlist), 'Embeddings and wordlist have different lengths in eval.py'
        word_2_embed_dict = { wordlist[i] : embeds[i] for i in range(len(embeds)) }
        results = eval_intrinsics(word_2_embed_dict)

    elif evaltype == 'synthetics':
        pass
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


# Evaluate similarity
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

# Evaluate analogy
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

def eval_intrinsics(word_vectors):
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
    print(all_tasks)
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

def eval_synthetics():
    pass


parser = argh.ArghParser()
parser.add_commands([eval_embeddings])

if __name__ == '__main__':
    parser.dispatch()
