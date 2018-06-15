# Small-Fry: Insanely compressed word embeddings

Small-Fry v. 0.1
----------------------

<img src="mascot.png" height="200" >


Table of Contents
-----------------

  * [Overview](#overview)  
  * [Installing from Source](#installing-from-source)
   * [Dependencies](#dependencies)
  * [Usage](#running-queries)
  * [Contact](#contact)

Overview
-----------------

Small-Fry is a general-purpose lossy compression algorithm for word embedding matrices. This is a research-prototype implementation of the algorithm. For many extrinsic inference tasks, Small-Fry can compress pre-existing source embeddings by 10-1000X, while preserving downstream performance to within <1% of the source embeddings. Please see TODO:paperlink for more details. 


Installing Small-Fry
-----------------
First, make sure your environment meets all of Small-Fry's [dependencies](#dependencies).

From source:



To install EmptyHeaded from source ensure that your system:

1. Meets all [dependencies](#dependencies) detailed below (or you are in our [Docker](#docker) contatiner)
2. Has [setup the EmptyHeaded environment](#setting-up-environment)
3. Has [compiled the QueryCompiler and Cython bindings](#compilation).

Dependencies
-----------------


Usage
-----------------
We provide demos of EmptyHeaded in iPython notebooks. 

We provide a tutorial of how to get started running your first EmptyHeaded query in the `docs/notebooks` folder. In this folder is a `Getting Started Tutorial` iPython notebook (amongst others) which can be loaded after executing `ipython notebook`.

The syntax for all queries run in [EmptyHeaded: A Relational Engine for Graph Processing](http://arxiv.org/abs/1503.02368) can be found in `docs/notebooks/graph` and the syntax for all queries run in [Old Techniques for New Join Algorithms: A Case Study in RDF Processing](http://arxiv.org/abs/1602.03557) can be found in `docs/notebooks/rdf` (pipelining for LUBM 8 is still a WIP being merged to master).

*A note on benchmarking:* the execution time for each query can be found with our timer that outputs to the shell with `Time[Query]`. Our timers seperate pure query execution time and from the time spent loading from disk for the user.

Contact
-----------------

[Christopher Aberger](http://web.stanford.edu/~caberger/)





This is a research prototype for Small-Fry, a word embeddings compression algorithm.

Recommended to use as API -- add modules to path and ```import smallfry as sfry```.  <br />
Also supports command line usage: <br />
Recommended cmd line usage for compression: <br />
```python smallfry.py compress source-path prior-path -m {MEM_BUDGET} --word-rep trie --write-word-rep ```<br />
Recommended cmd line usage for querying: <br />
```python smallfry.py query word word-representation-path sfry-path```<br />

Usage Notes: <br />
-- Small-Fry produces ```.sfry``` directories, which are the compressed representations of the word embeddings. <br />
-- If no specific memory budget is required, a bitrate of 1 is used by default, which generally works well. <br />
-- Small-Fry produced word representations are either Python dicts or marisa-tries. Alternatively, the user can be responsible for using the word list and handling the word -> index mapping separately (TODO). <br />
-- It is recommended to use Small-Fry programmatically. To query, use the ```sfry.load``` function, which returns a wrapper object for a memory mapped version of the compressed Small-Fry representations. The use the member function ```.query()``` on the wrapper object.

DEPENDENCIES:<br />
Numpy <br />
Scipy <br />
Scikit-learn <br />
Marisa-Trie <br />
Arghs <br />


