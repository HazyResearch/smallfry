#!/bin/bash

pushd ./src/smallfry/third_party/DrQA
pip install -r requirements.txt; python setup.py develop
popd