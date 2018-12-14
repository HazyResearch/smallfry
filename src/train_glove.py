import pathlib
import os
import logging
import sys
import time
import numpy as np
import utils

def main():
    utils.init('train')
    train()
    utils.save_to_json(utils.config, utils.get_filename('_train_config.json'))
    logging.info('Run complete. Exiting train.py main method')

def train():
    logging.info('Beginning training of embeddings')
    start = time.time()
    src_dir = utils.get_src_dir()
    vocab, cooc, _ = utils.get_corpus_info(utils.config['corpus'])
    output = utils.perform_command_local(
        'bash {}/train_glove.sh {} {} {} {} {} {} {} {}'.format(
            src_dir,
            '{}/third_party/GloVe/build'.format(src_dir), # builddir
            vocab, # vocab_file
            cooc, # cooccurence shuf file
            utils.get_filename('_embeds.txt'), # output file
            utils.config['embeddim'], # embed_dim
            utils.config['lr'], # lr
            utils.config['epochs'], # epochs
            utils.config['threads']  # threads
        )
    )
    logging.info(output)
    elapsed = time.time() - start
    logging.info('Finished training GloVe embeddings. It took {} min.'.format(elapsed/60))

if __name__ == '__main__':
    main()
