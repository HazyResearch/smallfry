[![Build Status](https://travis-ci.com/HazyResearch/smallfry.svg?token=DY2tqn6MMHmNqLqricH4&branch=master)](https://travis-ci.com/HazyResearch/smallfry)

# SmallFry
Word embedding is one of the key component of modern NLP models. As large vocabulary is often relevant to attain strong task performance, word embeddings can consume a large amount of memory during training and inference. 

**Smallfry is a word embedding compression algorithm based on uniform quantization with automatic clipping.** The compressed embedding can support **training NLP models and perform inference with fixed pre-trained embeddings**. It can significantly reduce the training and inference memory footprint for NLP models built on top of word embeddings.

Our pytorch implementation is a dropin replacement for the original pytorch embedding layer. It first _automatically clips the extreme values of the embeddings, and then compress the embedding with uniform quantization_.

## Installation
To install smallfry package and use the compressed embedding layer, please
```
git clone --recursive https://github.com/HazyResearch/smallfry_pytorch.git
cd smallfry_pytorch
pip install -e .
```
Our implementation is tested under Python 3.6 and PyTorch 1.0.

## Usage
### Directly initialize a compressed embedding layer
The initialization argument of smallfry embedding layer is the same as original [PyTorch embedding layers](https://pytorch.org/docs/stable/nn.html#embedding). The only additional required argument is ```nbit=<the precision of the quantized numbers> ```, specifying what numerical precision is desired for the underlying smallfry embedding representation. Currently we support 1, 2, 4, 8, 16, 32 bit representations.  

Smallfry compresses pre-trained embeddings for training and inference. During initialization, the pre-trained embedding values can be loaded via a ```torch.FloatTensor``` or via a file in GloVe format where everyline represent a word vector without a file header. For example,

```
from smallfry import QuantEmbedding

# init with existing tensor
embed_from_tensor = 
    QuantEmbedding(num_embeddings=1000,    # vocabulary size
                   embedding_dim=50,       # embedding dimensionality
                   _weight=<a PyTorch FloatTensor with shape = (100, 50)>, 
                   nbit=4)                 # the quantization precision

# init with embedding files
embed_from_file = 
    QuantEmbedding(num_embeddings=1000,    # vocabulary size
                   embedding_dim=50,       # embedding dimensionality
                   nbit=2,                 # the quantization precision
                   embedding_file=<a GloVe style embedding file with 1000 rows>) 
```
The input tensor and file can contain either quantized or unquantized values, the QuantEmbedding layer will automatically detect the quantize if unquantized values are given.

### Replace and existing embedding layer with a quantized embeding layer
Another important scenerio is that given an existing model, one may want to replace all the PyTorch embedding layers with QuantEmbedding layer in a batch. We provide a helper function to achieve this in the following example where the new QuantEmbedding layers will directly initialize by quantizing the values in the original embedding layers.

```
from smallfry import quantize_embed
...
quantize_embed(model,   # a model, i.e. an instance of PyTorch nn.Module,
               nbit=2,  # the quantization precision)
```

## Example
To present a full demonstration, we prepared an training example using smallfry QuantEmbedding layers. In this example, we train a LSTM-based [DrQA](https://github.com/facebookresearch/DrQA) model for reading comprehension on the [SQuAD1.1](https://rajpurkar.github.io/SQuAD-explorer/) dataset. We train the DrQA model on top of fixed pre-trained [GloVe](https://nlp.stanford.edu/projects/glove/) Embedding, using 2-bit quantization. 

### Setup
In order to prepare for training the DrQA model, please run
```
bash prepare_drqa.sh
```
This will automatically download the required processed data and install the DrQA package.

### Run
After the setup is done, the training can be launched via
```
bash run_drqa.sh
```
By the end of training, the original training embedding layer should achieve a ~73.86% dev set F1 score, which 2 bit smallfry quantized embedding achieves ~73.72%.
