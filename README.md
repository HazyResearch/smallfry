# Small-Fry: Insanely compressed word embeddings 


DISCLAIMER: This is a "research" implementation -- not robust. Contact tginart if bugs are found.

Recommended use as API. Command line usage:
Usage for compression: ```python src/smallfry.py compress {path} {priorpath} {R} {memorybudget}```<br />
Usage for querying: ```python src/smallfry.py query {word} {index_dict} {dim} {sfry_path}```<br />


TODOs:<br />
more robust debugging<br />
incomplete logging <br />
improve path I/O<br />
parafor the quantization <br />
fast k_means incomplete <br />
comment code<br />
coded and indexing compression functionality<br />
prior is cased<br />
and many, many more....<br />
