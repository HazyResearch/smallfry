import bitarray as ba
import numpy as np
from sklearn.cluster import KMeans

class Smallfry():

    def __init__(self, bit_arr, codebook, dim):
        self.bin_rep = bit_arr
        self.codebk = codebook
        self.dim = dim
            
    def _decode(self, embed_id):
        '''
        Internal helper script
        '''
        b = int(np.log2(len(self.codebk)))
        offset = embed_id*b*self.dim
        to_bit_arr = lambda i,b : ba.bitarray(bin(i)[2:].zfill(b))
        d = { self.codebk[i] : to_bit_arr(i,b) for i in range(2**b) }
        return self.bin_rep[offset:offset+b*self.dim].decode(d)

    def decode(self, idx_tensor):
        '''
        Decodes the binary representation, supporting tensorized indexing
        '''
        decode_embs = np.array([self._decode(i) for i in idx_tensor.flatten()])
        return decode_embs.reshape(idx_tensor.shape + (self.dim,))

    @staticmethod
    def quantize(embeddings,
                b=1,
                block_len=1,
                optimizer='iterative',
                max_iter=120,
                tol=0.01,
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
        return Smallfry(bit_arr, codebook, dim)

    @staticmethod
    def serialize(self, filepath):
        '''
        Serializes binary representation to file
        Includes metadata as {filepath}.meta
        '''
        bin_file = open(filepath, 'wb')
        meta_file = open(filepath+'.meta', 'w')

        self.bin_rep.tofile(bin_file)
        meta_file.write(json.dumps([self.codebk, self.dim]))

        bin_file.close()
        meta_file.close()

    @staticmethod
    def deserialize(self, filepath):
        '''
        Reads a Smallfry object
        '''
        bin_file = open(filepath,'rb')
        metadata_file = open(filepath+'.meta','r')

        bit_arr = ba.bitarray()
        bit_arr.fromfile(bin_file)
        metadata = json.loads(metedata_file.read())

        bin_file.close()
        metadata_file.close()

        return Smallfry(bit_arr, metadata[0], metadata[1])

