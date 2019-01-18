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
from third_party.sentence_classification.train_classifier import train_sentiment
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
    elif utils.config['evaltype'] == 'synthetics-large-dim':
        results = evaluate_synthetics_large_dim(utils.config['embedpath'])
    elif utils.config['evaltype'] == 'sentiment':
        results = evaluate_sentiment(
            utils.config['embedpath'],
            get_sentiment_data_path(),
            utils.config['compress-config']['seed'],
            utils.config['tunelr'],
            utils.config['dataset'],
            utils.config['epochs'],
            lr=utils.config['lr']
        )
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

def get_sentiment_data_path():
    return str(pathlib.PurePath(utils.get_src_dir(),
               'third_party', 'sentence_classification', 'data'))

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

    # Other
    results['embed-mean-euclidean-dist'] = np.mean(np.linalg.norm(base_embeds-embeds,axis=1))
    results['semantic-dist'] = np.mean([distance.cosine(embeds[i],base_embeds[i]) for i in range(len(embeds))])
    results['mean'] = np.mean(embeds)
    results['var'] = np.var(embeds)

    ### Covariance error (X^T X)
    compute_gram_or_cov_errors(embeds, base_embeds, False, 'cov', results)
    ### Gram matrix error (XX^T)
    compute_gram_or_cov_errors(embeds, base_embeds, True, 'gram', results)
    return results

def evaluate_synthetics_large_dim(embed_path):
    '''Evaluates synthetics'''
    embeds,_ = utils.load_embeddings(embed_path)
    results = {}
    # Compute delta's between compressed embedding matrix and large-dimensional
    # base embedding matrix (eg, compressed 50d glove400k kernel matrix vs.
    # uncompressed 300d glove400k kernel matrix)
    embedtype = utils.config['compress-config']['embedtype']
    if embedtype == 'glove400k':
        large_dim = 300
    elif embedtype == 'glove-wiki400k-am':
        large_dim = 400
    else:
        assert embedtype == 'fasttext1m'
        large_dim = 300

    base_path_large_dim,_ = utils.get_base_embed_info(embedtype, large_dim)
    base_embeds_large_dim,_ = utils.load_embeddings(base_path_large_dim)
    compute_gram_or_cov_errors(embeds, base_embeds_large_dim, True, 'gram-large-dim', results)
    return results

def compute_gram_or_cov_errors(embeds, base_embeds, use_gram, type_str, results):
    logging.info('Beginning compute_gram_or_cov_errors')
    if use_gram:
        n = 10000
        embeds = embeds[:n]
        base_embeds = base_embeds[:n]
        compressed = embeds @ embeds.T
        base = base_embeds @ base_embeds.T
    else:
        assert embeds.shape[1] == base_embeds.shape[1]
        compressed = embeds.T @ embeds
        base = base_embeds.T @ base_embeds

    # compute spectrum of base_embeds to extract minimum eigenvalue of X^T X
    logging.info('Beginning SVD of base_embeds')
    base_sing_vals = np.linalg.svd(base_embeds, compute_uv=False)
    base_eigs = base_sing_vals**2
    eig_min = base_eigs[-1]
    lambdas = [eig_min/100, eig_min/10, eig_min, eig_min*10, eig_min*100]

    # Frob error
    logging.info('Beginning Frobenius error computations')
    results[type_str + '-frob-error'] = np.linalg.norm(base-compressed)
    results[type_str + '-frob-norm'] = np.linalg.norm(compressed)
    results[type_str + '-base-frob-norm'] = np.linalg.norm(base)
    # Spec Error
    logging.info('Beginning spectral error computations')
    results[type_str + '-spec-error'] = np.linalg.norm(base-compressed, 2)
    results[type_str + '-spec-norm'] = np.linalg.norm(compressed,  2)
    results[type_str + '-base-spec-norm'] = np.linalg.norm(base, 2)
    # Delta1,Delta2
    logging.info('Beginning (Delta1,Delta2) computations')
    results[type_str + '-base-eig-min'] = eig_min
    results[type_str + '-lambdas'] = lambdas
    delta1_results = [0] * len(lambdas)
    delta2_results = [0] * len(lambdas)
    for i,lam in enumerate(lambdas):
        delta1_results[i], delta2_results[i], _ = utils.delta_approximation(base, compressed,  lambda_ = lam)
    results[type_str + '-delta1s'] = delta1_results
    results[type_str + '-delta2s'] = delta2_results
    logging.info('Finished compute_gram_or_cov_errors')

def evaluate_sentiment(embed_path, data_path, seed, tunelr, dataset, epochs, lr=-1):
    if tunelr:
        lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
        results = {}
        results['best-lr'] = 0
        results['best-val-err'] = 1.1 # error is between 0 and 1
        results['best-test-err'] = 1.1
        results['lrs'] = []
        results['val-errs'] = []
        results['test-errs'] = []
        for lr in lrs:
            cmdlines = ["--dataset", dataset, 
                        "--path", data_path + "/", 
                        "--embedding", embed_path, 
                        "--cnn", 
                        "--max_epoch", str(epochs), 
                        "--model_seed", str(seed), 
                        "--data_seed", str(seed),
                        "--lr", str(lr)]
            err_valid, err_test = train_sentiment(cmdlines)
            logging.info('lr: {}, accuracy (valid/test): {}/{}'.format(
                            lr, err_valid, err_test))
            results['lrs'].append(lr)
            results['val-errs'].append(err_valid)
            results['test-errs'].append(err_test)
            if err_valid < results['best-val-err']:
                results['best-lr'] = lr
                results['best-val-err'] = err_valid
                results['best-test-err'] = err_test
    else:
        assert lr > 0, 'Must specify positive learning rate'
        results = {}
        cmdlines = ["--dataset", dataset, 
                    "--path", data_path + "/", 
                    "--embedding", embed_path, 
                    "--cnn", 
                    "--max_epoch", str(epochs), 
                    "--model_seed", str(seed), 
                    "--data_seed", str(seed),
                    "--lr", str(lr)]
        err_valid, err_test = train_sentiment(cmdlines)
        logging.info('lr: {}, accuracy (valid/test): {}/{}'.format(
                lr, err_valid, err_test))
        results["val-err"] = err_valid
        results["test-err"] = err_test
    return results

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
