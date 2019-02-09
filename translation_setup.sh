#!/bin/bash

# prepare and preprocess for IWSLT14 German to English translation task
. /proj/anaconda3/etc/profile.d/conda.sh
conda create -n smallfry_pytorch1.0 python=3.6
conda activate smallfry_pytorch1.0
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
pip install torchtext
pip install torch torchvision
pip install scipy

# Install Smallfry
pip install -e .
pushd ../
export PYTHONPATH=$PYTHONPATH:$(pwd)
popd

echo smallfry installed

# Installs FairSeq
pushd ./src/third_party/low-memory-fnn-training/third_party/fairseq
pip install -r requirements.txt
python setup.py build develop
popd

echo fairseq installed

pushd ./src/third_party/DrQA
pip install -e .
popd

echo drqa installed

fairseq_dir=$(pwd)/src/third_party/low-memory-fnn-training/third_party/fairseq
app_dir=$(pwd)/src/third_party/low-memory-fnn-training/apps/fairseq
mkdir -p $app_dir/data-bin

cd $fairseq_dir/examples/translation
bash prepare-iwslt14.sh
cd ../..

echo iwslt14 data prepared

# Binarize the dataset
TEXT=examples/translation/iwslt14.tokenized.de-en
python preprocess.py --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir $app_dir/data-bin/iwslt14.tokenized.de-en

echo iwslt14 data processed




