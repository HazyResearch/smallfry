#!/bin/bash
BUILDDIR=$1 # third_party/GloVe/build
VOCAB_FILE=$2 # File containing vocabulary and truncated unigram counts (created by vocab_count script)
COOCCURRENCE_SHUF_FILE=$3 # shuffled cooccurence file.
OUTPUT_FILE=$4 # Filename, excluding extension, for word vector output.
EMBED_DIM=$5 # Dimension of word embeddings (Default: 50)
LR=$6 # Initial learning rate. (Default: 0.05)
EPOCHS=$7 # Number of training iterations (Default: 25)
THREADS=$8 # Number of threads to use (Default: 8)
VERBOSE=2
# Notable unspecified options: xmax, alpha, binary (default: save to txt file).

echo "$ $BUILDDIR/glove -vocab-file $VOCAB_FILE -input-file $COOCCURRENCE_SHUF_FILE -save-file $OUTPUT_FILE -vector-size $EMBED_DIM -eta $LR -iter $EPOCHS -threads $THREADS -verbose $VERBOSE"
$BUILDDIR/glove -vocab-file $VOCAB_FILE -input-file $COOCCURRENCE_SHUF_FILE -save-file $OUTPUT_FILE -vector-size $EMBED_DIM -eta $LR -iter $EPOCHS -threads $THREADS -verbose $VERBOSE
