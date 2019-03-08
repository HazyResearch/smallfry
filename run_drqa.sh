#!/bin/bash

# python -u ./src/smallfry/third_party/DrQA/scripts/reader/train.py --embed-dir= --embedding-file ./src/glove.6B.300d.txt --embedding-dim 300 --random-seed 1

python -u ./src/smallfry/third_party/DrQA/scripts/reader/train.py --embed-dir='' --embedding-file ./src/smallfry/third_party/DrQA/data/glove.6B.300d.txt --embedding-dim 300 --random-seed 1 --use-quant-embed --nbit 32 
