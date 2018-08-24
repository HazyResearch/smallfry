# Lloyd-Max Quantizer for Embeddings

<img src="/docs/mascot.png" height="200" >

Table of Contents
-----------------

  * [Overview](#overview)  
  * [Dependencies](#dependencies)
  * [Installing Small-Fry](#installing-small-fry)
  * [Usage](#usage)
  * [Contact](#contact)

Overview
-----------------

Lloyd-Max quantization is a classical approach for inferring locally optimal quantizations for data sets. In the context of clustering, it is known as the k-means algorithm. This lossy compression scheme blah blah.


Dependencies
-----------------
 runs on Python 3.6. All dependencies can be found in `requirements.txt`. They can be installed via:

`pip install -r requirements.txt`

Installing Small-Fry
-----------------
The easiest way to install Small-Fry is via pip install:

`pip install .`

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
