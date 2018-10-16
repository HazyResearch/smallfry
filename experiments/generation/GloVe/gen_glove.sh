#!/bin/bash
set -e

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

make

COOC_EXISTS=$1
CORPUS=$2 #text8
VOCAB_FILE=$3
COOCCURRENCE_FILE=$4
COOCCURRENCE_SHUF_FILE=$5
BUILDDIR=build
SAVE_FILE=${15}
VERBOSE=2
MEMORY=$6 #4
VOCAB_MIN_COUNT=${13} #5
VECTOR_SIZE=$7 #50
MAX_VOCAB=$8
MAX_ITER=$9
WINDOW_SIZE=${10} #15
BINARY=2
NUM_THREADS=${11} #32
X_MAX=10
SEED=${12}
ETA=${14}

echo
if [ $COOC_EXISTS = "0" ]; then 
    echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
    $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE -max-vocab $MAX_VOCAB < $CORPUS > $VOCAB_FILE
    echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
    $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
    echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE -seed $SEED < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
fi
echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE -seed $SEED -eta $ETA
