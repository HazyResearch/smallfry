#!/bin/bash

# NOTE: This only works if called from base conda environment!
printf "\n**** Updating apt-get and installing git-core. ****\n"
apt-get --assume-yes update
apt-get --assume-yes install git-core

# Execute the command that was passed in as a string to this script.
cmd="sudo docker run --runtime=nvidia -v /proj:/proj -it --rm tensorflow/tensorflow:latest-gpu-py3 apt-get --assume-yes update; apt-get --assume-yes install git-core; $1"
printf "\nExecute command: '$cmd'\n"
eval $cmd