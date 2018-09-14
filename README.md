This repository is the experimental testbench for the embeddings compression project.

For the Smallfry readme, see the smallfry directory.

This codebase is designed to run on AWS with Starcluster.

We assume a directory structure outside of the repository of the following sort:

- base directory: /proj/smallfry

- this repo should live in /proj/smallfry/git/smallfry

- we should have evaluation data sets in /proj/smallfry/embeddings_benchmark

- produced embeddings are stored in /proj/smallfry/embeddings

- and probably many more assumptions


NOTES:

- See maker.py for embeddings creation
- See eval.py for embeddings evaluation
- See merger.py for data merging 
- See plotter.py for plotting
