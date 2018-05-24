# small-fry


DISCLAIMER: not debugged, not robust, not good implementation

API Usage:

Compress: 
Inputs: path (filepath to text format embs), dim (# cols in mat), R (avg bitrate)
Outputs: word2idx(dict of words to rows -- user responsibility), sfry_path (where the compressed embeddings live)
example use: sfry.compress('glove.txt',300,1.2)
Notes: 
- Must have '.txt' ending
- automatically writes a bunch of files, including the 'glove.sfry' (compressed embeddings), word dicts, and inflated embeddings

Query: 
Inputs: word (the string you want to query for), word2idx (dict maps words to row id), dim (# cols in mat), sfry_path (path to compressed embeddings returned by compress)
Outputs: r (inflated row)
example use: r = sfry.query('election', word2idx, dim, sfry_path)
Notes:
- word2idx is written to file by compress 
- sfry_path holds codebook, metadata, and binary embeddings representations
- Ideally, there will be an "initialize" call that returns an actual query wrapper which has pre-loaded the 'glove.sfry' into a tmpfile in memory. Then all queries can be applied through this wrapper object -- TODO


TODOs:
autodetect dimension
codebk and allots stored by compressor
improve path I/O
parafor the quantization as threads
clean out imports
comment code
load compressed files as temp files for query
build pyobj wrapper for querying the embeddings
coded and indexing compression functionality
ask wrapper for memory footprint
prior is cased
and many, many more....
