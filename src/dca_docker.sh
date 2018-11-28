#!/bin/bash

# Launch docker container, then install git (to get git-hash and git-diff),
# and launch the command that was passed in as a string to this script.
cmd="sudo docker run --runtime=nvidia -v /proj:/proj -it --rm tensorflow/tensorflow:latest-gpu-py3 /bin/bash -c \"apt-get --assume-yes update; apt-get --assume-yes install git-core; $1\""
printf "Execute the following command in docker container: '$cmd'\n"
eval $cmd