import bitarray as ba
import numpy as np
import marisa_trie as marisa

class Smallfry():

    def __init__(self, bit_arr, dim, codebook, words):
        self.bin_rep = bit_arr
        self.codebk = codebook
        self.dim = dim
        self.wordtrie = marisa.Trie(words) if words != None else None

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

    def query(self, w):
        '''
        Queries for word w from the binary representation. 
        Requires initialization with word trie. 
        '''
        assert self.words != None, 'Words must be provided to use query'
        OOV = not w in self.wordtrie
        return np.ones(self.dim)/self.dim if OOV else _decode(self.wordtrie[w])

    def serialize(self, filepath):
        '''
        Serializes binary representation to file
        Includes metadata as {filepath}.meta
        '''
        bin_file = open(filepath, 'wb')
        meta_file = open(filepath+'.meta', 'w')

        self.bin_rep.tofile(bin_file)
        meta_file.write(json.dumps([self.dim, self.codebk]))

        bin_file.close()
        meta_file.close()

    def deserialize(self, filepath):
        '''
        Reads a Smallfry object
        '''
        bin_file = open(filepath,'rb')
        metadata_file = open(filepath+'.meta','r')

        bit_arr = ba.bitarray()
        bit_arr.fromfile(bin_file)

        return Smallfry

