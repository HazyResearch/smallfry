'''
GENERAL PURPOSE EMBEDDINGS EVALUATION SCRIPT
'''
import os
import re
import time
import logging
import pathlib
import numpy as np
import subprocess
from scipy.spatial import distance
from third_party.hyperwords.hyperwords import ws_eval, analogy_eval
from third_party.hyperwords.hyperwords.representations.embedding import Embedding
from third_party.DrQA.scripts.reader.train import train_drqa
import utils

def main():
    utils.init('evaluate')
    evaluate_embeds()
    utils.save_to_json(utils.config, utils.get_filename('_final.json'))
    logging.info('Run complete. Exiting evaluate.py main method')

def evaluate_embeds():
    logging.info('Beginning evaluation')
    start = time.time()
    if utils.config['evaltype'] == 'qa':
        results = evaluate_qa(
            utils.config['embedpath'],
            utils.config['compress-config']['embeddim'],
            utils.config['compress-config']['seed']
        )
    elif utils.config['evaltype'] == 'intrinsics':
        results = evaluate_intrinsics(utils.config['embedpath'])
    elif utils.config['evaltype'] == 'synthetics':
        results = evaluate_synthetics(utils.config['embedpath'])
    elapsed = time.time() - start
    results['elapsed'] = elapsed
    utils.config['results'] = results
    logging.info('Finished evaluating embeddings. It took {} min.'.format(elapsed/60))

def evaluate_qa(embed_path, embed_dim, seed):
    qa_args = ['--embed-dir=', '--embedding-file', embed_path,
               '--embedding-dim', str(embed_dim), '--random-seed', str(seed)]
    f1_scores,exact_match_scores = train_drqa(qa_args, use_cuda=utils.config['cuda'])
    results = {}
    results['f1-scores'] = f1_scores
    results['exact-match-scores'] = exact_match_scores
    results['best-f1'] = max(f1_scores)
    results['best-exact-match'] = max(exact_match_scores)
    logging.info('DrQA Results: best-f1 = {}, best-exact-match = {}'.format(
        results['best-f1'], results['best-exact-match']))
    return results

def evaluate_intrinsics(embed_path):
    '''Evaluates intrinsics benchmarks'''
    embeds, wordlist = utils.load_embeddings(embed_path)
    embed_dict = { wordlist[i] : embeds[i] for i in range(len(embeds)) }

    similarity_tasks = [
            "bruni_men",
            "luong_rare",
            "radinsky_mturk",
            "simlex999",
            "ws353",
            "ws353_relatedness",
            "ws353_similarity"
    ]
    analogy_tasks = [
            "google",
            "msr"
    ]

    results = {}
    similarity_results = []
    analogy_results = []
    for task in similarity_tasks:
        task_path = get_task_path('ws', task)
        output = evaluate_similarity(embed_dict, task_path)
        results[task] = output
        similarity_results.append(output)
    for task in analogy_tasks:
        task_path = get_task_path('analogy', task)
        output = evaluate_analogy(embed_dict, task_path)
        results[task + "-add"] = output[0]
        results[task + "-mul"] = output[1]
        analogy_results.extend(output)
    results['analogy-avg-score'] = np.mean(analogy_results)
    results['similarity-avg-score'] = np.mean(similarity_results)

    logging.info('Word similarity and analogy results:')
    for task,score in results.items():
        logging.info('\t{}: {:.4f}'.format(task, score))
    return results

# Evaluate analogy -- ROUTINE WRITTEN BY MAXLAM
# -----------------------------------------
# embed_dict - dictionary where keys are words, values are word vectors.
# task_path - path to similarity dataset
# return - similarity score
def evaluate_analogy(embed_dict, task_path):
    print("Evaluating analogy: %s" % task_path)
    assert os.path.exists(task_path)
    data = analogy_eval.read_test_set(task_path)
    xi, ix = analogy_eval.get_vocab(data)
    representation = BootstrapEmbeddings(embed_dict)
    return analogy_eval.evaluate(representation, data, xi, ix)

# Evaluate similarity -- ROUTINE WRITTEN BY MAXLAM
# -----------------------------------------
# embed_dict - dictionary where keys are words, values are word vectors.
# task_path - path to similarity dataset
# return - similarity score
def evaluate_similarity(embed_dict, task_path):
    '''Evaluates sim intrinsic suite'''
    print("Evaluating similarity: %s" % task_path)
    assert os.path.exists(task_path)
    data = ws_eval.read_test_set(task_path)
    representation = BootstrapEmbeddings(embed_dict)
    return ws_eval.evaluate(representation, data)

def get_task_path(task_type, task_name):
    return str(pathlib.PurePath(utils.get_src_dir(),
               'third_party', 'hyperwords', 'testsets',
               task_type, task_name + '.txt'))

def evaluate_synthetics(embed_path):
    '''Evaluates synthetics'''
    embeds,_ = utils.load_embeddings(embed_path)
    base_embeds,_ = utils.load_embeddings(
        utils.config['compress-config']['base-embed-path'])

    results = {}
    ### Embedding error (X)
    # Frob error (X)
    results['embed-frob-error'] = np.linalg.norm(base_embeds-embeds)
    results['embed-frob-norm'] = np.linalg.norm(embeds)
    results['base-embed-frob-norm'] = np.linalg.norm(base_embeds)
    # Spec Error (X)
    results['embed-spec-error'] = np.linalg.norm(base_embeds-embeds,2)
    results['embed-spec-norm'] = np.linalg.norm(embeds,2)
    results['base-embed-spec-norm'] = np.linalg.norm(base_embeds,2)

    ### Covariance error (X^T X)
    compute_gram_or_cov_errors(embeds, base_embeds, False, results)
    ### Gram matrix error (XX^T)
    compute_gram_or_cov_errors(embeds, base_embeds, True, results)

    # Other
    results['embed-mean-euclidean-dist'] = np.mean(np.linalg.norm(base_embeds-embeds,axis=1))
    results['semantic-dist'] = np.mean([distance.cosine(embeds[i],base_embeds[i]) for i in range(len(embeds))])
    results['mean'] = np.mean(embeds)
    results['var'] = np.var(embeds)

    return results

def compute_gram_or_cov_errors(embeds, base_embeds, use_gram, results):
    if use_gram:
        n = 10000
        embeds = embeds[:n]
        base_embeds = base_embeds[:n]
        compressed = embeds @ embeds.T
        base = base_embeds @ base_embeds.T
        type_str = 'gram'
    else:
        compressed = embeds.T @ embeds
        base = base_embeds.T @ base_embeds
        type_str = 'cov'

    # compute spectrum of base_embeds to extract minimum eigenvalue of X^T X
    base_sing_vals = np.linalg.svd(base_embeds, compute_uv=False)
    base_eigs = base_sing_vals**2
    eig_min = base_eigs[-1]
    lambdas = [eig_min/100, eig_min/10, eig_min, eig_min*10, eig_min*100]

    # Frob error
    results[type_str + '-frob-error'] = np.linalg.norm(base-compressed)
    results[type_str + '-frob-norm'] = np.linalg.norm(compressed)
    results[type_str + '-base-frob-norm'] = np.linalg.norm(base)
    # Spec Error
    results[type_str + '-spec-error'] = np.linalg.norm(base-compressed, 2)
    results[type_str + '-spec-norm'] = np.linalg.norm(compressed,  2)
    results[type_str + '-base-spec-norm'] = np.linalg.norm(base, 2)
    # Delta1,Delta2
    results[type_str + '-base-eig-min'] = eig_min
    results[type_str + '-lambdas'] = lambdas
    delta1_results = [0] * len(lambdas)
    delta2_results = [0] * len(lambdas)
    for i,lam in enumerate(lambdas):
        delta1_results[i], delta2_results[i], _ = utils.delta_approximation(base, compressed,  lambda_ = lam)
    results[type_str + '-delta1s'] = delta1_results
    results[type_str + '-delta2s'] = delta2_results

# def perform_command_local(command):
#     ''' performs a command -- author: MAXLAM'''
#     out = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True).decode('utf-8')
#     return out

# def eval_sent(embed_txt_path, seed, dataset=None):
#     '''
#     Sentiment analysis evaluation using Senwu's code -- supports perceptron, CNN, and LSTM models with various datasets
#     '''
#     def parse_senwu_outlogs(outlog):
#         lines = outlog.split('\n')
#         return float(lines[-3].split(' ')[-1])

#     logging.info('starting sentiment')
#     models = ['lstm', 'cnn', 'la']
#     datasets = ['mr',
#                 'subj', 
#                 'cr', 
#                 'sst', 
#                 'trec', 
#                 'mpqa'] if dataset == 'all' else [dataset]
#     res = dict()
#     for model in models:
#         for dataset in datasets:
#             command = "python2  %s --dataset %s --path %s --embedding %s --cv 0 --%s --out %s" % (
#                 str(pathlib.PurePath(get_senwu_sentiment_directory(),'train_classifier.py')),
#                 dataset,
#                 get_harvardnlp_sentiment_data_directory(),
#                 embed_txt_path,
#                 model,
#                 get_senwu_sentiment_out_directory()
#             ) 
#             cmd_output_txt = perform_command_local(command)
#             logging.info(cmd_output_txt)
#             res['sentiment-score-%s-%s'%(model,dataset)] = parse_senwu_outlogs(cmd_output_txt)
#     logging.info('done with sentiment evals')
#     return res

class BootstrapEmbeddings(Embedding):
    def __init__(self, embed_dict, normalize=True):
        self.dim = len(list(embed_dict.values())[0])
        self.m = np.stack(list(embed_dict.values()))
        self.iw = {i:k for i,k in enumerate(embed_dict.keys())}
        self.wi = {k:i for i,k in self.iw.items()}
        if normalize:
            self.normalize()

if __name__ == '__main__':
    main()
