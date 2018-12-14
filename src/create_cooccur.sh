#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

BUILDDIR=$1 # third_party/GloVe/build
CORPUS=$2
VOCAB_FILE=$3
COOCCURRENCE_FILE=$4
COOCCURRENCE_SHUF_FILE=$5
MAX_VOCAB=$6
MEMORY=$7
MIN_COUNT=1
VERBOSE=2
# Notable unspecified options: max-product, window-size

echo "$ $BUILDDIR/vocab_count -max-vocab $MAX_VOCAB -min-count $MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -max-vocab $MAX_VOCAB -min-count $MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
echo "$ $BUILDDIR/cooccur -vocab-file $VOCAB_FILE -memory $MEMORY -verbose $VERBOSE < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -vocab-file $VOCAB_FILE -memory $MEMORY -verbose $VERBOSE < $CORPUS > $COOCCURRENCE_FILE
echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
