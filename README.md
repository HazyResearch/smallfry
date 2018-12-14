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