#!/bin/bash
set -e

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

make

CORPUS=$1 #text8
VOCAB_FILE=$1.vocab.txt
COOCCURRENCE_FILE=$1.cooccurrence.bin
COOCCURRENCE_SHUF_FILE=$1.cooccurrence.shuf.bin
BUILDDIR=build
VERBOSE=2
MEMORY=$5 #4
VOCAB_MIN_COUNT=5
VECTOR_SIZE=$2 #50
MAX_VOCAB=$3
MAX_ITER=$6
WINDOW_SIZE=$7 #15
BINARY=2
NUM_THREADS=$4 #32
X_MAX=10
SEED=$8

echo
echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE -max-vocab $MAX_VOCAB < $CORPUS > $VOCAB_FILE
echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -seed $SEED < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -seed $SEED< $CORPUS > $COOCCURRENCE_FILE
