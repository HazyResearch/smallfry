[![Build Status](https://travis-ci.com/HazyResearch/smallfry.svg?token=DY2tqn6MMHmNqLqricH4&branch=master)](https://travis-ci.com/HazyResearch/smallfry)

# SmallFry
We provide an implementation of the SmallFry algorithm for compressing
and evaluating embeddings.  We also implement a number of baselines,
including k-means quantization and deep compositional autoencoders (DCA).
We provide code for evaluating these embeddings on word similarity and
analogy tasks, as well as for evaluating the performance of the DrQA
model trained on the compressed embeddings.

To clone this repository:

```bash
$ git clone --recursive https://github.com/HazyResearch/smallfry.git
```

Submodules related:
In order to recursively clone the submodules in the smallfry repo, you might need access to the following private repo:
```
https://github.com/HazyResearch/fairseq-fork
https://github.com/avnermay/sentence_classification
```
Please reach out to [Nimit Sohoni](https://github.com/nimz) for accessing the fairse-fork repo and [Avner May](https://github.com/avnermay) for accessing the sentence-classification repo.


Installation
(We support python 3.6 and pytorch 1.0)
git clone --recursive https://github.com/HazyResearch/smallfry.git
cd smallfry
git submodule update --init --recursive
pip install -e .
