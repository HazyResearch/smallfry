#!/bin/bash

# Launch docker container, then install git (to get git-hash and git-diff),
# and launch the command that was passed in as a string to this script.
sudo apt-get --assume-yes install nvidia-384 nvidia-modprobe
cmd="sudo docker run --runtime=nvidia -v /proj:/proj --rm tensorflow/tensorflow:latest-gpu-py3 /bin/bash -c \"apt-get --assume-yes update; apt-get --assume-yes install git-core; . /proj/anaconda3/etc/profile.d/conda.sh; conda activate smallfry; conda env list; which python; $1\""
printf "Execute the following command in docker container: '$cmd'\n"
eval $cmd