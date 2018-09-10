import bitarray as ba
import numpy as np
import json
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

class Smallfry(nn.Module):

    def __init__(self, bin_rep, codebook, dim):
        super(Smallfry, self).__init__()
        self.bin_rep = bin_rep
        #check that codebook length is a power of 2
        assert not (len(codebook) & (len(codebook)-1)),'Codes not a power of 2'
        self.codebook = codebook
        self.dim = dim


    def forward(self, input):
        orig_device = input.device
        embed_query = torch.from_numpy(self.decode(
            input.to(device='cpu').numpy())).to(device=orig_device)
        embed_query.requires_grad = False
        return embed_query

    def decode(self, idx_tensor):
        '''
        Decodes the binary representation, supporting tensorized indexing
        '''
        decode_embs = np.array([self._decode(i) for i in idx_tensor.flatten()])
        print(decode_embs)
        return decode_embs.reshape(idx_tensor.shape + (self.dim,))

    def _decode(self, embed_id):
        '''
        Internal helper script
        '''
        b = int(np.log2(len(self.codebook)))
        l = len(self.codebook[0])
        offset = embed_id*b*self.dim
        d = {self.codebook[i] : self._generate_bin(i,b) for i in range(2**b)}
        print(int(b*self.dim/l))
        return self.bin_rep[offset:offset+int(b*self.dim/l)].decode(d)

    @staticmethod
    def _generate_bin(i,b):
        '''
        helper
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

