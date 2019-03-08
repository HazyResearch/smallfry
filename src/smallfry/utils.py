import os
import sys
import socket
import json
import datetime
import logging
import pathlib
import time
import random
import subprocess
import argparse
import numpy as np
import getpass


def load_embeddings(path):
    """
    Loads a GloVe or FastText format embedding at specified path. Returns a
    vector of strings that represents the vocabulary and a 2-D numpy matrix that
    is the embeddings.
    """
    logging.info('Beginning to load embeddings')
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        wordlist = []
        embeddings = []
        if is_fasttext_format(lines): lines = lines[1:]
        for line in lines:
            row = line.strip('\n').split(' ')
            wordlist.append(row.pop(0))
            embeddings.append([float(i) for i in row])
        embeddings = np.array(embeddings)
    assert len(wordlist) == embeddings.shape[0], 'Embedding dim must match wordlist length.'
    logging.info('Finished loading embeddings')
    return embeddings, wordlist

def get_embedding_dimension(embed_path):
    with open(embed_path) as f_embed:
        for line in f_embed:
            if not is_fasttext_format([line]):
                pieces = line.rstrip().split(' ')
                embed_dim = len(pieces) - 1
                logging.info('Loading ' + str(embed_dim) +
                                ' dimensional embedding')
                break
    assert embed_dim > 0
    return embed_dim

def is_fasttext_format(lines):
    first_line = lines[0].strip('\n').split(' ')
    return len(first_line) == 2 and first_line[0].isdigit() and first_line[1].isdigit()