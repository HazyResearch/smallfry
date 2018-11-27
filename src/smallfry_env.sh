#!/bin/bash

# NOTE: This only works if called from base conda environment!


printf "\nnvcc --version (BEFORE)\n"
nvcc --version

printf "\nmodinfo nvidia (BEFORE)\n"
modinfo nvidia | grep "^version:" | sed 's/^version: *//;'


# Make sure conda and cuda are in path
export PATH=/usr/local/cuda/bin:/proj/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

printf "\nnvcc --version (AFTER)\n"
nvcc --version

printf "\nmodinfo nvidia (AFTER)\n"
modinfo nvidia | grep "^version:" | sed 's/^version: *//;'

# Activate smallfry env
printf "source activate smallfry\n"
source activate smallfry

printf "\nnvcc --version (AFTER2)\n"
nvcc --version

printf "\nmodinfo nvidia (AFTER2)\n"
modinfo nvidia | grep "^version:" | sed 's/^version: *//;'

printf "\nconda env list\n"
conda env list

printf "which python\n"
which python

# Execute the command that was passed in as a string to this script.
printf "\nExecute command: '$1'\n"
eval $1
