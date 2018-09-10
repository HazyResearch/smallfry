import bitarray as ba
import numpy as np
import json
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from functools import reduce

class Smallfry(nn.Module):

    def __init__(self, bin_rep, codebook, dim):
        super(Smallfry, self).__init__()
        self.bin_rep = bin_rep
        assert int(np.log2(len(codebook))) == np.log2(len(codebook)),\
            'Number of centroids (entries of codebook) must be a power of 2'
        self.codebook = codebook
        self.dim = dim
        self.bits_per_block = int(np.log2(len(self.codebook)))
        self.block_len = len(codebook[0])
        assert self.dim % self.block_len == 0, 'Block len must divide dim'
        self.num_blocks =  int(self.dim / self.block_len)
        self.decode_dict = \
            {self.codebook[i] : self._generate_bin(i, self.bits_per_block) 
                for i in range(2**self.bits_per_block)}

    def forward(self, input):
        orig_device = input.device
        embed_query = torch.from_numpy(self.decode(
            input.to(device='cpu').numpy())).to(device=orig_device)
        embed_query.requires_grad = False
        return embed_query

    def decode_all_in_one(self, index_tensor):
        '''
        Decodes the binary representation, supporting tensorized indexing
        '''
        flat_index = index_tensor.flatten()
        decoded_embs = np.zeros(flat_index.shape + (self.dim,))
        embed_length = self.bits_per_block * self.num_blocks
        for (i,embed_index) in enumerate(index_tensor.flatten()):
            for j in range(self.num_blocks):
                offset = embed_index * embed_length + j * self.bits_per_block
                inflated_block = self.bin_rep[offset:offset + self.bits_per_block].decode(
                        self.decode_dict
                    )
                decoded_embs[i,j*self.block_len:(j+1)*self.block_len] = \
                    inflated_block[0]                   
        return decoded_embs.reshape(index_tensor.shape + (self.dim,))

    def decode(self, idx_tensor):
        '''
        Decodes the binary representation, supporting tensorized indexing
        '''
        decode_embs = np.array([self._decode(i) for i in idx_tensor.flatten()])
        print(decode_embs)
        return decode_embs.reshape(idx_tensor.shape + (self.dim,))

    def _decode(self, embed_id):
        '''
            Decode a single embedding.
        '''
        embed_length = self.bits_per_block * self.num_blocks
        offset = embed_id * embed_length
        tuples = self.bin_rep[offset:offset + embed_length].decode(self.decode_dict)
        concatenated_tuples = reduce((lambda x, y: x + y), tuples)
        return np.array(concatenated_tuples)

    @staticmethod
    def _generate_bin(i,b):
        '''
        Represents the integer i in b bits
        '''
        return ba.bitarray(bin(i)[2:].zfill(b))

    @staticmethod
    def quantize(embeddings,
                b=1,
                block_len=1,
                optimizer='iterative',
                max_iter=70,
                n_init=1,
                tol=0.01,
                r_seed=1234
                ):
        '''
        This method applies the Lloyd-Max quantizer with specified block dimension.
        The bitrate is specified PER BLOCK.
        '''
        v,dim = embeddings.shape
        assert dim % block_len == 0, 'Block len must divide the embedding dim'
        kmeans = KMeans(n_clusters=2**b, max_iter=max_iter, n_init=n_init, tol=tol, random_state=r_seed)
        kmeans = kmeans.fit(embeddings.reshape(int(v*dim/block_len), block_len))
        bin_rep = ba.bitarray()
        d = {i : Smallfry._generate_bin(i,b) for i in range(2**b)}
        bin_rep.encode(d, kmeans.labels_)
        codebook = [tuple(centroid) for centroid in kmeans.cluster_centers_]
        return Smallfry(bin_rep, codebook, dim)

    @staticmethod
    def serialize(sfry, filepath):
        '''
        Serializes binary representation to file
        Includes metadata as {filepath}.meta
        '''
        with open(filepath, 'wb') as bin_file:
            sfry.bin_rep.tofile(bin_file)
        with open(filepath+'.meta','w') as meta_file:
            meta_file.write(json.dumps([sfry.codebook, sfry.dim]))

    @staticmethod
    def deserialize(filepath):
        '''
        Reads in a Smallfry object
        '''
        bin_rep = ba.bitarray()
        metadata = None
        with open(filepath, 'rb') as bin_file:
            bin_rep.fromfile(bin_file)
        with open(filepath+'.meta','r') as meta_file:
            metadata = json.loads(meta_file.read())
        #json converts tuples to lists, so need to undo this
        codebook = [tuple(code) for code in metadata[0]]
        return Smallfry(bin_rep, codebook, metadata[1])
