import argh
import os
import pathlib
import sys
from git import Repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) #FOR LOCAL IMPORTS
from experimental_utils import * 

def setup_testbench_drive():
    dirs = [get_base_embed_path_head(),
            get_base_outputdir(),
            get_launch_path(),
            get_qsub_log_path(),
            get_plots_path(),
            get_corpus_path()]
    for directory in dirs:
        print(f"making dir {directory}")
        os.makedirs(directory)
        if 'launches' in directory or 'qsub' in directory:
            os.chdir(directory)
            for subdir in ['gen','maker','eval']:
            os.mkdir(subdir)

    
    benchmarks_path = str(
        pathlib.PurePath(get_base_directory(),'embeddings_benchmark'))
    os.makedirs(benchmarks_path)
    #os.chdir(benchmarks_path)
    repos = [('https://github.com/facebookresearch/DrQA.git',get_drqa_directory()),
            ('https://github.com/SenWu/sentence_classification.git',get_senwu_sentiment_directory()),
            ('https://github.com/yuhaozhang/tacred-relation.git',get_relation_directory()),
            ('https://github.com/harvardnlp/sent-conv-torch.git',get_harvardnlp_sentiment_data_directory())]
    for repo in repos:
        print(f"cloning repo {repo[0]}")
        Repo.clone_from(repo[0],repo[1])


parser = argh.ArghParser()
parser.add_commands([setup_testbench_drive])

if __name__ == '__main__':
    parser.dispatch()