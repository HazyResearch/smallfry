#!/bin/bash

# NOTE: This only works if called from base conda environment!

# Make sure conda is in path
# export PATH=$PATH:/proj/anaconda3/bin

# This is necessary for conda activate to work properly
. /proj/anaconda3/etc/profile.d/conda.sh

# Activate smallfry env
printf "conda activate smallfry\n"
conda activate smallfry

printf "\nconda env list\n"
conda env list

printf "which python\n"
which python

# Execute the command that was passed in as a string to this script.
printf "\nExecute command: '$1'\n"
eval $1
