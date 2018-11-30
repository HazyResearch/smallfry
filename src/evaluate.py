'''
GENERAL PURPOSE EMBEDDINGS EVALUATION SCRIPT
'''
import os
import re
import time
import logging
import numpy as np
import subprocess
from scipy.spatial import distance
# from third_party.hyperwords.hyperwords import ws_eval, analogy_eval
# from third_party.hyperwords.hyperwords.representations.embedding import BootstrapEmbeddings
from third_party.DrQA.scripts.reader.train import train_drqa
import utils

def main():
    utils.init('evaluate')
    logging.info('Beginning evaluation.')
    evaluate_embeds()
    utils.save_dict_as_json(utils.config, utils.get_filename('_final.json'))
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
    logging.info('Finished evaluating embeddings. It took {} min.'.format(elapsed/60))
    results['elapsed'] = elapsed
    utils.config['results'] = results

def evaluate_qa(embed_path, embed_dim, seed):
    qa_args = ['--embed-dir=', '--embedding-file', embed_path,
               '--embedding-dim', str(embed_dim), '--random-seed', str(seed)]
    f1_scores,exact_match_scores = train_drqa(qa_args, use_cuda=utils.config['cuda'])
    results = {}
    results['f1-scores'] = f1_scores
    results['exact-match-scores'] = exact_match_scores
    results['best-f1'] = max(f1_scores)
    results['best-exact-match'] = max(exact_match_scores)
    return results

def evaluate_intrinsics(embed_path):
    '''Evaluates intrinsics benchmarks'''

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
    embeds, wordlist = utils.load_embeddings(embed_path)
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
            logging.info("%s not in list of similarity or analogy tasks." % os.path.basename(task_path))
        partial_result = "%s - %s\n" % (os.path.basename(task_path), str(output))
        results += partial_result
        logging.info(partial_result)

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
    logging.info('======= Begin intrinsic results ========')
    logging.info(results)
    logging.info('======== End intrinsic results =========')
    return results_dict

def evaluate_synthetics(embed_path):
    '''Evaluates synthetics'''
    embeds,_ = utils.load_embeddings(embed_path)
    base_embeds,_ = utils.load_embeddings(
        utils.config['compress-config']['base-embed-path'])

    # TODO FILL THIS IN
    results = {}
    results['embed-frob-dist'] = np.linalg.norm(base_embeds-embeds)
    results['embed-frob-norm'] = np.linalg.norm(embeds)
    results['mean'] = np.mean(embeds)
    results['var'] = np.var(embeds)
    results['embed-mean-euclidean-dist'] = np.mean(np.linalg.norm(base_embeds-embeds,axis=1))
    results['semantic-dist'] = np.mean([distance.cosine(embeds[i],base_embeds[i]) for i in range(len(embeds))])
    return results

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

if __name__ == '__main__':
    main()
