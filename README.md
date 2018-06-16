# Small-Fry: Insanely compressed word embeddings

<img src="/docs/mascot.png" height="200" >

Table of Contents
-----------------

  * [Overview](#overview)  
  * [Installing Small-Fry](#installing-small-fry)
   * [Dependencies](#dependencies)
  * [Usage](#usage)
  * [Contact](#contact)

Overview
-----------------

Small-Fry is a general-purpose lossy compression algorithm for word embedding matrices. This is a research-prototype implementation of the algorithm. The algorithm is simple two-stage scheme, combining a Lloyd-Max quantizer with an arithemtic source code. It's compressed embeddings achieve state-of-the-art rate-performance across extrinisic NLP tasks. Small-Fry uses variable-precision, quantizing more frequent word vectors with higher fidelity.

For many extrinsic inference tasks, Small-Fry can compress pre-existing source embeddings by 10-1000X, while preserving downstream performance to within <1% of the source embeddings.


Installing Small-Fry
-----------------
First, make sure your environment meets all of Small-Fry's [dependencies](#dependencies).

`pip install .`


Dependencies
-----------------
All dependencies can be found in `requirements.txt`. They can be installed via:

`pip install -r requirements.txt`


Usage
-----------------

Small-Fry is provided as an API, for a simple demo using the Small-Fry API, see `examples`. 

Some notes:
* Small-Fry produces ```.sfry``` directories, which are the compressed representations of the word embeddings. <br />
* If no specific memory budget is required, a bitrate of 1 is used by default, which generally works well. <br />
* Small-Fry produced word representations are either Python dicts or marisa-tries. Alternatively, the user can be responsible for using the word list and handling the word -> index mapping separately (TODO). <br />
* It is recommended to use Small-Fry programmatically. To query, use the ```sfry.load``` function, which returns a wrapper object for a memory mapped version of the compressed Small-Fry representations. The use the member function ```.query()``` on the wrapper object.


Contact
-----------------

[Tony Ginart](http://web.stanford.edu/~tginart/)
