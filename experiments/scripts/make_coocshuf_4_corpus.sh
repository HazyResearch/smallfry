#!/bin/bash
set -e

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

make

CORPUS=$1 #text8
VOCAB_FILE=${CORPUS}.vocab.txt
BUILDDIR=build
VERBOSE=2
MEMORY=$2 #4
VOCAB_MIN_COUNT=5 #5
MAX_VOCAB=$3
WINDOW_SIZE=$4 #15
NUM_THREADS=$5 #32
SEED=$6
PATH=${CORPUS}.maxvocab_${MAX_VOCAB}.windowsize_${WINDOW_SIZE}.seed_${SEED}.vocabmincount_${VOCAB_MIN_COUNT}.memory_${MEMORY}
COOCCURRENCE_FILE=${PATH}.cooccurrence.bin
COOCCURRENCE_SHUF_FILE=${PATH}.cooccurrence.shuf.bin

echo
echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE -max-vocab $MAX_VOCAB < $CORPUS > $VOCAB_FILE
echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -seed $SEED < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -seed $SEED< $CORPUS > $COOCCURRENCE_FILE
echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE