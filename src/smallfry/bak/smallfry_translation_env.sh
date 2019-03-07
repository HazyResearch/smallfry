#!/bin/bash

# NOTE: This only works if called from base conda environment!

# Make sure conda is in path
# export PATH=$PATH:/proj/anaconda3/bin

# This is necessary for conda activate to work properly
. /proj/anaconda3/etc/profile.d/conda.sh

# Activate smallfry env
printf "conda activate smallfry_translation\n"
conda activate smallfry_pytorch1.0

# export smallfry path
export PYTHONPATH=$PYTHONPATH:/proj/smallfry/git

printf "\nconda env list\n"
conda env list

printf "which python\n"
which python

# Execute the command that was passed in as a string to this script.
printf "\nExecute command: '$1'\n"
eval $1
