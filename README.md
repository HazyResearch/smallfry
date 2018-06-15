# Small-Fry: Insanely compressed word embeddings

Small-Fry v. 0.1: Insanely compressed word embeddings
----------------------

<img src="mascot.png" height="200" >


Table of Contents
-----------------

  * [Overview](#overview)  
  * [Installing from Source](#installing-from-source)
   * [Dependencies](#dependencies)
  * [Usage](#running-queries)
  * [Contact](#contact)

Overview
-----------------

Small-Fry is a general-purpose lossy compression algorithm for word embedding matrices.

Installing from Source
-----------------
To install EmptyHeaded from source ensure that your system:

1. Meets all [dependencies](#dependencies) detailed below (or you are in our [Docker](#docker) contatiner)
2. Has [setup the EmptyHeaded environment](#setting-up-environment)
3. Has [compiled the QueryCompiler and Cython bindings](#compilation).

Dependencies
-----------------
Behind the scenes a lot goes on in our engine. This is no walk-in-the-park library-based engine--we have advanced theoretical compilation techniques, code generation, and highly optimized code for hardware--all spanning multiple programming languages. As such we have a fair number of dependencies. Try using our [Docker images](#docker) where everything is taken care of for you already.

**AVX**

A fundamental dependency of our system is that it is designed for machines that support the Advanced Vector Extensions (AVX) instruction set which is standard in modern and future hardware generations. Our performance is highly dependent on this instruction set being available. We currently DO NOT support old hardware generations without AVX. 

* Mac or Linux operating system
* GCC 5.3 (Linux) or Apple LLVM version 7.0.2 (Mac)
* clang-format
* C++11 
* cmake 2.8 or higher (C++)
* jemalloc (C++)
* tbb (C++, Intel)
* sbt (scala)
* iPython Notebook (python)
* cython (python)
* jpype 0.6.1 (python)
* pandas (python)

The instructions below briefly describe some of our dependencies and why we have them. A complete list of our dependencies as well as how to install them is in our `Dockerfile`. Note: we provide JPype and an install script in our `dependencies` folder.

**Why iPython Notebook?**

iPython Notebook provides a easy and user-friendly front-end for users to enter, compile, and run queries.

**Why clang-format?**

EmptyHeaded generates code from a high level datalog description. Making generated code look nice is a challenging task! [Clang-format](http://clang.llvm.org/docs/ClangFormat.html) is an easy solution.

**Why jemalloc?**

The GNU malloc is ineffecient for multi-threaded programs. [jemalloc](https://www.facebook.com/notes/facebook-engineering/scalable-memory-allocation-using-jemalloc/480222803919/) to the rescue!

**Why TBB?**

Writing an efficient parallel-sort is a challenging task. Why re-invent the wheel? Use [Intel's TBB](https://www.threadingbuildingblocks.org/).

**Why Pandas?**

[Pandas DataFrames](http://pandas.pydata.org/pandas-docs/stable/dsintro.html) provides a nice and highly used front-end for EmptyHeaded to accept tables from. We can also run without DataFrames but who doesn't love DataFrames?

**Why cython?**

[Cython](http://cython.org/) is awesome! It enables an easy bridge from C++ to Python, also it is just a great way to speed up Python code.

**Why JPype?**

JPype is our bridge between python and java. We provide this one in our `dependencies` folder along with a simple install script.

Docker
-----------------
Make your life easier and use our [Docker images](https://hub.docker.com/r/craberger/emptyheaded/) which are *always* up to date. 

Unfortunately iPython notebooks and Docker containers do not interact easily, but you can run standard python scripts just fine in these containers! 

Two easy ways to get started in a container:

1. Simply inspect our iPython notebook tutorials in this repository (can view on github) and make the corresponding python programs. 
2. Checkout our python test scripts in the `test` folder, `./test/testAll.sh` kick them it all off.

Setting up Environment
-----------------

```
source env.sh
```

EmptyHeaded relies on two environment variables being set.

-`EMPTYHEADED_HOME` the root directory for the EmptyHeaded project

-`EMPTYHEADED_HOME/python` must be in the python search path

The easiest way to meet these dependencies is to run `source env.sh` provided in the root of this repository. Note: This script will set the `PYTHON_PATH` variable.

Compilation
-----------------

```
./compile.sh
```

The compiler needs to be compiled (which makes me wonder [who](http://homepage.ntlworld.com/edmund.grimley-evans/bcompiler.html) compiled the first compiler?). This compiles our Scala code and a few static cython bindings.

Running Queries
-----------------
We provide demos of EmptyHeaded in iPython notebooks. 

We provide a tutorial of how to get started running your first EmptyHeaded query in the `docs/notebooks` folder. In this folder is a `Getting Started Tutorial` iPython notebook (amongst others) which can be loaded after executing `ipython notebook`.

The syntax for all queries run in [EmptyHeaded: A Relational Engine for Graph Processing](http://arxiv.org/abs/1503.02368) can be found in `docs/notebooks/graph` and the syntax for all queries run in [Old Techniques for New Join Algorithms: A Case Study in RDF Processing](http://arxiv.org/abs/1602.03557) can be found in `docs/notebooks/rdf` (pipelining for LUBM 8 is still a WIP being merged to master).

*A note on benchmarking:* the execution time for each query can be found with our timer that outputs to the shell with `Time[Query]`. Our timers seperate pure query execution time and from the time spent loading from disk for the user.

Contact
-----------------

[Christopher Aberger](http://web.stanford.edu/~caberger/)





This is a research prototype for Small-Fry, a word embeddings compression algorithm.

Recommended to use as API -- add modules to path and ```import smallfry as sfry```.  <br />
Also supports command line usage: <br />
Recommended cmd line usage for compression: <br />
```python smallfry.py compress source-path prior-path -m {MEM_BUDGET} --word-rep trie --write-word-rep ```<br />
Recommended cmd line usage for querying: <br />
```python smallfry.py query word word-representation-path sfry-path```<br />

Usage Notes: <br />
-- Small-Fry produces ```.sfry``` directories, which are the compressed representations of the word embeddings. <br />
-- If no specific memory budget is required, a bitrate of 1 is used by default, which generally works well. <br />
-- Small-Fry produced word representations are either Python dicts or marisa-tries. Alternatively, the user can be responsible for using the word list and handling the word -> index mapping separately (TODO). <br />
-- It is recommended to use Small-Fry programmatically. To query, use the ```sfry.load``` function, which returns a wrapper object for a memory mapped version of the compressed Small-Fry representations. The use the member function ```.query()``` on the wrapper object.

DEPENDENCIES:<br />
Numpy <br />
Scipy <br />
Scikit-learn <br />
Marisa-Trie <br />
Arghs <br />


