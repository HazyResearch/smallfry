# Small-Fry: Insanely compressed word embeddings

Small-Fry v. 0.1
----------------------

<img src="mascot.png" height="200" >


Table of Contents
-----------------

  * [Overview](#overview)  
  * [Installing Small-Fry](#installing small-fry)
   * [Dependencies](#dependencies)
  * [Usage](#usage)
  * [Contact](#contact)

Overview
-----------------

Small-Fry is a general-purpose lossy compression algorithm for word embedding matrices. This is a research-prototype implementation of the algorithm. The algorithm is simple two-stage scheme, combining a Lloyd-Max quantizer with an arithemtic source code. It's compressed embeddings achieve state-of-the-art rate-performance across extrinisic NLP tasks. Small-Fry uses variable-precision, quantizing more frequent word vectors with higher fidelity.

For many extrinsic inference tasks, Small-Fry can compress pre-existing source embeddings by 10-1000X, while preserving downstream performance to within <1% of the source embeddings. Please see TODO:paperlink for more details. 


Installing Small-Fry
-----------------
First, make sure your environment meets all of Small-Fry's [dependencies](#dependencies).

`pip install .`


Dependencies
-----------------
All dependencies can be found in `requirements.txt`.

They can be installed via:

`pip install -r requirements.txt`


Usage
-----------------

We recommend using Small-Fry as an API, however command line interfaces are provided.

For API documentations see `docs`. For a simple demo using the Small-Fry API, see `examples`. 

For direct command line use TODO


Contact
-----------------

[Tony Ginart](http://web.stanford.edu/~tginart/)

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


