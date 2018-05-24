# small-fry


DISCLAIMER: not debugged, not robust, not good implementation

API Usage:

Compress: 
Inputs: path (filepath to text format embs), dim (# cols in mat), R (avg bitrate) <br />
Outputs: word2idx(dict of words to rows -- user responsibility), sfry_path (where the compressed embeddings live) <br />
example use: sfry.compress('glove.txt',300,1.2)<br />
Notes: <br />
- Must have '.txt' ending <br />
- automatically writes a bunch of files, including the 'glove.sfry' (compressed embeddings), word dicts, and inflated embeddings <br />

Query: <br />
Inputs: word (the string you want to query for), word2idx (dict maps words to row id), dim (# cols in mat), sfry_path (path to compressed embeddings returned by compress)<br />
Outputs: r (inflated row)<br />
example use: r = sfry.query('election', word2idx, dim, sfry_path)<br />
Notes:<br />
- word2idx is written to file by compress <br />
- sfry_path holds codebook, metadata, and binary embeddings representations<br />
- Ideally, there will be an "initialize" call that returns an actual query wrapper which has pre-loaded the 'glove.sfry' into a tmpfile in memory. Then all queries can be applied through this wrapper object -- TODO<br />


TODOs:<br />
autodetect dimension<br />
codebk and allots stored by compressor<br />
improve path I/O<br />
parafor the quantization as threads<br />
clean out imports<br />
comment code<br />
load compressed files as temp files for query<br />
build pyobj wrapper for querying the embeddings<br />
coded and indexing compression functionality<br />
ask wrapper for memory footprint<br />
prior is cased<br />
and many, many more....<br />
