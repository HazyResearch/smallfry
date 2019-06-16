#!/bin/bash

pushd ./src/smallfry/third_party/DrQA
pip install -r requirements.txt; python setup.py develop
mkdir data
popd

pushd ./src/smallfry/third_party/DrQA/data
wget https://www.dropbox.com/s/tdekiyr8r14ogtk/datasets.zip?dl=0
unzip datasets.zip
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
popd
