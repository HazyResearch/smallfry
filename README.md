# Small-Fry: Insanely compressed word embeddings 


DISCLAIMER: This is a "research" implementation -- not robust. Contact tginart if bugs are found.

Recommended to use as API -- add modules to path and ```import smallfry as sfry```.  <br />
Also supports command line usage: <br />
Recommended cmd line usage for compression: <br />
```python smallfry.py compress source-path prior-path -m {MEM_BUDGET} --word-rep trie --write-word-rep ```<br />
Recommended cmd line usage for querying: <br />
```python smallfry.py query word word-representation-path sfry-path```<br />

Usage Notes: <br />
If no specific memory budget is required, a bitrate of 1 is used by default, which generally works well. <br />
Word representations are either Python dicts or marisa-tries. 

DEPENDENCIES:<br />
Numpy <br />
Scipy <br />
Scikit-learn <br />
Marisa-Trie <br />

TODOs:<br />
more robust debugging<br />
incomplete logging <br />
improve path I/O<br />
parafor the quantization <br />
fast k_means incomplete <br />
cleaner comments<br />
coded and indexing compression functionality<br />
prior is cased<br />
and many, many more....<br />
