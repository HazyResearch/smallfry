#!/bin/bash

# NOTE: This only works if called from base conda environment!

# Make sure conda is in path
export PATH="/proj/anaconda3/bin:$PATH"

# Activate PyKernels env
printf "source activate smallfry\n"
source activate PyKernels

printf "\nconda env list\n"
conda env list

printf "which python\n"
which python

# Execute the command that was passed in as a string to this script.
printf "\nExecute command: '$1'\n"
eval $1