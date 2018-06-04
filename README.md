# Small-Fry: Insanely compressed word embeddings 

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


