import numpy as np
import bitarray as ba
from sklearn.cluster import KMeans
from .smallfry import Smallfry

def quantize(embeddings,
                b=1,
                block_len=1,
                optimizer='iterative',
                max_iter=120,
                tol=0.01,
                words=None
                ):
    '''
    This method applies the Lloyd-Max quantizer with specified block dimension.
    The bitrate is specified PER BLOCK.
    '''
    v,dim = embeddings.shape
    assert dim % block_len == 0, 'Block len must divide the embedding dim'
    kmeans = KMeans(n_clusters=2**b, max_iter=max_iter, tol=tol)
    kmeans = kmeans.fit(embeddings.reshape(int(v*dim/block_len), block_len))
    bit_arr = ba.bitarray()
    d = [(i, ba.bitarray(bin(i)[2:].zfill(b))) for i in range(2**b)]
    bit_arr.encode(dict(d), kmeans.labels_)
    codebook = [tuple(centroid) for centroid in kmeans.cluster_centers_]
    return Smallfry(bit_arr, codebook, dim, words)
