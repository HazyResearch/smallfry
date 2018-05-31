# small-fry


DISCLAIMER: Not robust. Contact tginart if bugs are found.

Usage for compression: ```python src/smallfry.py compress {path} {priorpath} {R} {memorybudget}```
Usage for querying: ```python src/smallfry.py query {word} {index_dict} {dim} {sfry_path}```

Can also be used as API. Import the module and call the functions.

Notes:<br />
- word2idx is written to file by compress <br />
- sfry_path holds codebook, metadata, and binary embeddings representations<br />
- Ideally, there will be an "initialize" call that returns an actual query wrapper which has pre-loaded the 'glove.sfry' into a tmpfile in memory. Then all queries can be applied through this wrapper object -- TODO<br />

TODOs:<br />
more robust debugging
autodetect dimension<br />
improve path I/O<br />
parafor the quantization <br />
fast k_means is questionable <br />
clean out imports<br />
comment code<br />
load compressed files as temp files for query<br />
build pyobj wrapper for querying the embeddings<br />
coded and indexing compression functionality<br />
ask wrapper for memory footprint<br />
prior is cased<br />
and many, many more....<br />
