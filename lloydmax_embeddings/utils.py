import numpy as np
import json
import bitarray as ba

def serialize(bit_arr, metadata, out_path)
    '''
    Writes a lloyd-max quantized binary and metadata to a directory
    '''
    lmqbin_filepath = out_path + '.lmqbin'
    metadata_filepath = out_path + '.meta'

    lmqbin_file = open(lmqbin_filepath,'wb')
    metadata_file = open(metadata_filepath,'w')

    bit_arr.tofile(lmqbin_file)
    metadata_file.write(json.dumps(metadata))

def deserialize(lmqbin_filepath, metadata_filepath):
    '''
    Loads a lloyd-max quantized binary and the metadata from file
    '''

    lmqbin_file = open(lmqbin_filepath,'rb')
    metadata_file = open(metadata_filepath,'r')

    bit_arr = ba.bitarray()
    bit_arr.fromfile(lmqbin_file)

    return bit_arr, json.loads(metadata_file.read())


