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
import logging
import numpy as np
from subprocess import check_output
from scipy.spatial import distance
from hyperwords import ws_eval, analogy_eval
from hyperwords.representations.embedding import *
from smallfry.utils import load_embeddings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) #hacky way to import experimental_utils
from experimental_utils import *

#TODO: Change from argh to argparse?

def eval_embeddings(embed_path, evaltype, seed=None, epochs=None, dataset=None):
    '''
    This is the front-end routine for experimental evaluation. 
    For each acceptable experiment type, denoted with 'evaltype', it dispatches
    the appropriate evaluation subroutine.
    Finally, results are written to file.
    As of Sept. 16, valid 'evaltype' selections are: 'QA' OR 'intrinsics' OR 'synthetics'
    NOTE: 'embed_path' refers to the TOP-LEVEL embedding directory path, NOT the path to the .txt embeddings file
    '''
    embed_name = os.path.basename(embed_path)
    log_path_head = str(pathlib.PurePath(embed_path,embed_name))
    log_path = '%s_%s-eval.log' % (log_path_head, evaltype) 
    init_logging(log_path)
    results = None
    logging.info('Evaltype confirmed: %s' % evaltype)
    if do_results_already_exist(embed_path, evaltype):
        logging.info("OOPS these results already are present -- ABORTING")

    # determine evaltype and send off to that subroutine -- SEE THIS LOGIC TREE FOR VALID EVALTYPES
    if evaltype == 'QA':
        seed = int(seed)
        if epochs == None:
            epochs = 50
        results = eval_qa(fetch_embeds_txt_path(embed_path), fetch_dim(embed_path), seed, epochs, qa_log_path='%s-tmp'%log_path)
    elif evaltype == 'intrinsics':
        results = eval_intrinsics(embed_path)
    elif evaltype == 'synthetics':
        results = eval_synthetics(embed_path)
    elif evaltype == 'sentiment':
        seed = int(seed)
        results = eval_sent(fetch_embeds_txt_path(embed_path), seed, dataset)
    else:
        assert False, 'bad evaltype given to eval()'

    #wrap up the results and document stuff
    results['githash-%s' % evaltype] = get_git_hash()
    results['seed-%s' % evaltype] = seed
    logging.info("Evaluation complete! Writing results to file... ")
    results_to_file(embed_path, evaltype, results)

'''
CORE EVALUATION ROUTINES =======================
a new routine must be added for each evaltype!
'''

def eval_qa(word_vectors_path, dim, seed, epochs, qa_log_path="", finetune_top_k=0, extra_args=""):
    '''Calls DrQA's training routine'''

    #to_dict: transforms QA output into results-style json dict
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
        return result

    # Evaluate on the word vectors
    cd_dir = "cd %s" % get_drqa_directory()
    eval_print("Writing intermediate training to output path: %s" % qa_log_path)
    
    # WARNING: REALLY DANGEROUS SINCE MAKES ASSUMPTIONS ABOUT 
    # FILEPATHS AND THEIR EXISTENCE
    #TODO WARNING FIX EPOCHS HERE
    python_command = "python scripts/reader/train.py --random-seed %d --embedding-dim %d  --embed-dir=  --embedding-file %s  --num-epochs %s --tune-partial %d %s 2>&1 | tee %s" % (seed, dim, word_vectors_path, epochs, finetune_top_k, extra_args, qa_log_path)
    full_command = " && ".join([cd_dir, python_command])
    logging.info("Executing: %s" % full_command)
    text = perform_command_local(full_command)
    #eval_print("==============================") 
    logging.info("==============================")
    #eval_print("Output of DrQA run:")
    logging.info("Output of DrQA run:")
    #eval_print("==============================")
    logging.info("==============================")
    #eval_print(text)
    logging.info(text)
    #eval_print("==============================")
    logging.info("==============================")
    #eval_print("End output of DrQA run")
    logging.info("End output of DrQA run")
    #eval_print("==============================")
    logging.info("==============================")
    return to_dict(text)

def eval_intrinsics(embed_path):
    '''Evaluates intrinsics benchmarks given embed_path'''

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

    # Evaluate similarity -- ROUTINE WRITTEN BY MAXLAM
    # -----------------------------------------
    # word_vectors - dictionary where keys are words, values are word vectors.
    # task_path - path to similarity dataset
    # return - similarity score
    def evaluate_similarity(word_vectors, task_path):
        '''Evaluates sim intrinsic suite'''
        print("Evaluating similarity: %s" % task_path)
        assert os.path.exists(task_path)
        data = ws_eval.read_test_set(task_path)
        representation = BootstrapEmbeddings(word_vectors)
        return ws_eval.evaluate(representation, data)

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
    path_to_tasks = [os.path.dirname(os.path.abspath(__file__)) + "/testsets/" + x for x in all_tasks]

    results = ""
    results_dict = {}
    ana_score_sum = 0
    sim_score_sum = 0
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
            ana_score_sum += (output[0] + output[1])
        else:
            results_dict[task_name] = output
            if task_name != 'luong_rare': #we leave out rare words intrinsic
                sim_score_sum += output

    results_dict['analogy-avg-score'] = ana_score_sum/4 #four analogy tasks avg together
    results_dict['similarity-avg-score'] = sim_score_sum/5 #five sim tasks avg together        
    #eval_print("Results:")
    logging.info("Results:")
    #eval_print("------------------------------")
    logging.info("------------------------------")        
    #eval_print(results)
    logging.info(results)
    #eval_print("------------------------------")
    logging.info("------------------------------")
    return results_dict

def eval_synthetics(embed_path):
    '''Evaluates synthetics'''
    #TODO: what synthetics will we put in here?
    embeds, wordlist = fetch_embeds_4_eval(embed_path)
    base_embeds, base_wordlist = load_embeddings(fetch_base_embed_path(embed_path))

    res_rtn = dict()
    res_rtn['embed-fro-dist'] = np.linalg.norm(base_embeds-embeds)
    res_rtn['embed-fro-norm'] = np.linalg.norm(embeds)
    res_rtn['mean'] = np.mean(embeds)
    res_rtn['var'] = np.var(embeds)
    res_rtn['embed-mean-euclidean-dist'] = np.mean(np.linalg.norm(base_embeds-embeds,axis=1))
    res_rtn['semantic-dist'] = np.mean([distance.cosine(embeds[i],base_embeds[i]) for i in range(len(embeds))])
    return res_rtn

def eval_sent(embed_txt_path, seed, dataset=None):
    #TODO: sent eval not operational yet

    def parse_senwu_outlogs(outlog):
        lines = outlog.split('\n')
        return float(lines[-3].split(' ')[-1])

    logging.info('starting sentiment')
    models = ['lstm', 'cnn', 'la']
    datasets = ['mr', 'subj', 'cr', 'sst', 'trec', 'mpqa']
    if dataset != None:
        datasets = [dataset]
    res = dict()
    for model in models:
        for dataset in datasets:
            command = "python2  %s --dataset %s --path %s --embedding %s --cv 0 --%s --out %s" % (
                str(pathlib.PurePath(get_senwu_sentiment_directory(),'train_classifier.py')),
                dataset,
                get_harvardnlp_sentiment_data_directory(),
                embed_txt_path,
                model,
                get_senwu_sentiment_out_directory()
            ) 
            cmd_output_txt = perform_command_local(command)
            logging.info(cmd_output_txt)
            res['sentiment-score-%s-%s'%(model,dataset)] = parse_senwu_outlogs(cmd_output_txt)
    logging.info('done with sentiment evals')
    return res

parser = argh.ArghParser()
parser.add_commands([eval_embeddings])

if __name__ == '__main__':
    parser.dispatch()
