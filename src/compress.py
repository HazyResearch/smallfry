import os
import logging
import math
import time
import pathlib
import numpy as np
try:
    from sklearn.cluster import KMeans
    from third_party.neuralcompressor.nncompress.embed_compress import EmbeddingCompressor
except ImportError:
    pass
import utils

def main():
    utils.init('compress')
    utils.config['base-embed-path'], utils.config['vocab'] = get_embed_info()
    base_embeds,wordlist = utils.load_embeddings(utils.config['base-embed-path'])
    store_embed_memory_info(*base_embeds.shape)
    compress_and_save_embeddings(base_embeds, wordlist, utils.config['bitrate'])
    utils.save_dict_as_json(utils.config, utils.get_filename('_final.json'))
    logging.info('Run complete. Exiting compress.py main method.')

def store_embed_memory_info(v,d):
    assert d == utils.config['embeddim'], 'Embed dims do not match.'
    assert v == utils.config['vocab'], 'Vocab sizes do not match.'
    utils.config['exact-memory'] = get_exact_memory()
    utils.config['exact-compression-ratio'] = 32 * v * d / utils.config['exact-memory']
    utils.config['exact-bitrate'] = utils.config['exact-memory'] / (v * d)
    assert np.abs(utils.config['exact-bitrate'] - utils.config['bitrate']) < .01, \
           'Discrepency between exact and intended bitrates is >= 0.01.'

def get_embed_info():
    '''Get path to embedding, and size of vocab for embedding'''
    path = ''
    if utils.config['embedtype'] == 'glove400k':
        path_format_str = str(pathlib.PurePath(utils.config['basedir'],
                              'base_embeddings', 'glove400k', 'glove.6B.{}d.txt'))
        path = path_format_str.format(utils.config['embeddim'])
        vocab = 400000
    if utils.config['embedtype'] == 'glove10k':
        path_format_str = str(pathlib.PurePath(utils.config['basedir'],
                              'base_embeddings', 'glove10k', 'glove.6B.{}d.10k.txt'))
        path = path_format_str.format(utils.config['embeddim'])
        vocab = 10000
    return path,vocab

def compress_and_save_embeddings(X, wordlist, bit_rate):
    logging.info('Beggining to make embeddings')
    results = {}
    if utils.config['compresstype'] == 'uniform':
        Xq, frob_squared_error, elapsed = compress_uniform(X, bit_rate,
            adaptive_range=utils.config['adaptive'],
            stochastic_round=utils.config['stoch'],
            skip_quantize=utils.config['skipquant'])
    elif utils.config['compresstype'] == 'kmeans':
        Xq, frob_squared_error, elapsed = compress_kmeans(X, bit_rate,
            random_seed=utils.config['seed'])
    elif utils.config['compresstype'] == 'dca':
        work_dir = str(pathlib.PurePath(utils.config['rundir'],'dca_tmp'))
        Xq, frob_squared_error, elapsed, results_per_epoch = compress_dca(
            X, bit_rate, k=utils.config['k'], work_dir=work_dir,
            learning_rate=utils.config['lr'], batch_size=utils.config['batchsize'],
            grad_clip=utils.config['gradclip'], tau=utils.config['tau']
        )
        results['dca-results-per-epoch'] = results_per_epoch
    elif utils.config['compresstype'] == 'nocompress':
        Xq = X
        frob_squared_error = 0
        elapsed = 0
    results['frob-squared-error'] = frob_squared_error
    results['elapsed'] = elapsed
    utils.config['results'] = results
    utils.config['compressed-embed-path'] = utils.get_filename('_compressed_embeds.txt')
    utils.save_embeddings(utils.config['compressed-embed-path'], Xq, wordlist)
    logging.info('Finished making embeddings. It took {} min.'.format(elapsed/60))

def compress_kmeans(X, bit_rate, random_seed=None, n_init=1):
    # Tony's params for k-means: max_iter=70, n_init=1, tol=0.01.
    # default k-means params: max_iter=300, n_init=10, tol=1e-4
    kmeans = KMeans(n_clusters=2**bit_rate, random_state=random_seed, n_init=n_init)
    start = time.time()
    kmeans = kmeans.fit(X.reshape(X.size,1))
    elapsed = time.time() - start
    # map each element of X to the nearest centroid
    Xq = kmeans.cluster_centers_[kmeans.labels_].reshape(X.shape)
    frob_squared_error = kmeans.inertia_
    return Xq, frob_squared_error, elapsed

# def __init__(self, n_codebooks, n_centroids, model_path,
#         learning_rate=0.0001, batch_size=64, grad_clip=0.001, tau=1.0): # Avner change

def compress_dca(X, bit_rate, k=2, work_dir=os.getcwd(),
        learning_rate=0.0001, batch_size=64, grad_clip=0.001, tau=1.0):
    # TODO: Test inflate_dca_embeddings
    (v,d) = X.shape
    m = compute_m_dca(k, v, d, bit_rate)
    start = time.time()
    compressor = EmbeddingCompressor(m, k, work_dir,
        learning_rate=learning_rate, batch_size=batch_size,
        grad_clip=grad_clip, tau=tau)
    results_per_epoch = compressor.train(X)
    elapsed = time.time() - start
    _,frob_squared_error = compressor.evaluate(X)
    codes, codebook = compressor.export(X, work_dir)
    codes = np.array(codes)
    # reshape codebook as (m,k,d), representing m codebooks, each of size k x d.
    codebook = np.array(codebook).reshape(m,k,d)
    Xq = inflate_dca_embeddings(codes, codebook, m, k, v, d)
    return Xq, frob_squared_error, elapsed, results_per_epoch

def compress_uniform(X, bit_rate, adaptive_range=False, stochastic_round=False,
        skip_quantize=False):
    start = time.time()
    if adaptive_range:
        # TODO: Should we pass through stochastic_round here?
        range_limit = find_optimal_range(X, bit_rate, stochastic_round=False)
    else:
        range_limit = get_max_abs(X)

    Xq = _compress_uniform(X, bit_rate, range_limit,
        stochastic_round=stochastic_round, skip_quantize=skip_quantize)
    elapsed = time.time() - start
    frob_squared_error = np.linalg.norm(X-Xq)**2
    return Xq, frob_squared_error, elapsed

# Internal function.  This one expects an explicit range_limit.
def _compress_uniform(X, bit_rate, range_limit, stochastic_round=False,
        skip_quantize=False):
    '''
    Internal uniform quantization function (ADD MORE DESCRIPTION)
    '''
    assert range_limit >= 0, 'range_limit must be non-negative.'
    assert X.dtype == np.float or X.dtype == np.float64,\
                'Only floating point inputs allowed.'
    Xq = np.copy(X)
    if get_max_abs(Xq) > range_limit:
        np.clip(Xq, -range_limit, range_limit, out=Xq)
    if not skip_quantize and range_limit != 0:
        # We only need to quantize if skip_quantize is not set to true,
        # and range_limit != 0 (range_limit == 0 means the whole matrix is 
        # already set to 0)
        if bit_rate == 0:
            Xq[:] = 0
        elif bit_rate < 32:
            # affine transform to put Xq in [0,2**bit_rate - 1]
            Xq = (2**bit_rate - 1) * (Xq + range_limit) / (2 * range_limit)
            if stochastic_round:
                # each entry will round down if noise > fraction part
                np.ceil(Xq - np.random.rand(*Xq.shape), out=Xq)
            else:
                np.round(Xq, out=Xq)
            # undo affine transformation
            Xq = (Xq * 2 * range_limit) / (2**bit_rate - 1) - range_limit
        elif bit_rate >= 32:
            pass # don't quantize if bitrate >= 32
    return Xq

def find_optimal_range(X, bit_rate, stochastic_round=False, tol=1e-2):
    '''
    Find the best value to use to clip the embeddings before using uniform quantization.
    '''
    f = lambda range_limit : compress_and_compute_frob_squared_error(
        X, bit_rate, range_limit, stochastic_round=stochastic_round)

    return golden_section_search(f, 0, get_max_abs(X), tol=tol)

def compress_and_compute_frob_squared_error(X, bit_rate, range_limit, stochastic_round=False):
    '''
    Function which computes frob squared error after compression.  This function
    is used in the find_optimal_range function to find best clip value for
    adaptive range uniform compression.
    '''
    Xq = _compress_uniform(X, bit_rate, range_limit, stochastic_round=stochastic_round)
    return np.linalg.norm(X - Xq)**2

# TESTED
def golden_section_search(f, x_min, x_max, tol=1e-2):
    '''
    Find argmin of f between x_min and x_max (for f uni-modal).
    
    This function uses the golden-section search algorithm.
    It always maintains a list of four points [x1,x2,x3,x4],
    which are always spaced as: [a,a+(c^2)h,a+ch,a+h].
    for c = (math.sqrt(5) - 1) / 2 = 0.618...
    (c is equal to 1/phi, where phi = (1+sqrt(5))/2 is the golden ratio).
    The algorithm progressively reduces the size of the interval being
    considered by checking whether f(x2) < f(x3), and eliminating one of the
    endpoints accordingly; x4 is eliminated if f(x2) < f(x3), and x1 
    is eliminated otherwise.
    
    If f(a+(c^2)h) < f(a+ch), the new interval becomes
    >>> [a,a+(c^3)h,a+(c^2)h,a+ch] = [a,a+(c^2)(ch),a+c(ch),a+ch]
    (So h' = ch, a' = a)
    Otherwise, the interval becomes
    >>> [a',a'+(c^2)h',a'+ch', a'+h'], for a' = a+(c^2)h and h'=(h-(c^2)h)
    It is easy to check that a'+(c^2)h' = a + ch, and that a'+h' = a+h,
    So this interval is equal to [a+(c^2)h, a+ch, X, a+h], for X=a'+ch'
    The algorithm terminates when it has been narrowed
    down that the argmin must be in an interval of size < tol.
    '''
    # Initialize points
    # c is equal to 1/phi, for phi = (1+sqrt(5))/2
    c = (math.sqrt(5) - 1) / 2
    x1 = x_min
    x4 = x_max
    f_x1 = f(x1)
    f_x4 = f(x4)
    x2 = x1 + (x4-x1) * c**2
    x3 = x1 + (x4-x1) * c
    f_x2 = f(x2)
    f_x3 = f(x3)
    while (x4-x1 > tol):
        assert (math.isclose(x2, x1 + (x4 - x1) * c**2) and
                math.isclose(x3, x1 + (x4 - x1) * c))
        if f_x2 < f_x3:
            # The new points become [x1, NEW, x2, x3]
            x4,f_x4 = x3,f_x3
            x3,f_x3 = x2,f_x2
            x2 = x1 + (x4-x1) * c**2
            f_x2 = f(x2)
        else:
            # The new points become [x2, x3, NEW, x4]
            x1,f_x1 = x2,f_x2
            x2,f_x2 = x3,f_x3
            x3 = x1 + (x4-x1) * c
            f_x3 = f(x3)
        
    # Return x-value with minimum f(x) which was found.
    i = np.argmin([f_x1,f_x2,f_x3,f_x4])
    x = [x1,x2,x3,x4]
    return x[i]

# TESTED
def get_max_abs(X):
    return np.max(np.abs(X))

def inflate_dca_embeddings(codes, codebook, m, k, v, d):
    '''Inflates DCA output embeddings.
    'codes' is a v x m numpy array, where entry i,j corresponds to the index
    of the entry in the j'th codebook used by word i.
    'codebook' is a m x k x d numpy array, where codebook[i,:,:] represents
    the i'th codebook (of dimension k x d).
    To inflate the embeddings, we add up the m d-dimensional vectors
    (one from each codebook) used by each word.
    '''
    assert codes.shape == (v,m)
    assert codebook.shape == (m,k,d)
    embeds = np.zeros([v,d])
    for i in range(v):
        for j in range(m):
            embeds[i,:] += codebook[j, codes[i,j], :]
    return embeds

def compute_m_dca(k, v, d, bit_rate):
    return int(np.round(bit_rate * v * d / (v * np.log2(k) + 32 * d * k)))

def get_exact_memory():
    v = utils.config['vocab']
    d = utils.config['embeddim']
    bit_rate = utils.config['bitrate']
    if utils.config['compresstype'] == 'kmeans':
        num_centroids = 2**bit_rate
        mem = v * d * bit_rate + num_centroids * 32
    elif utils.config['compresstype'] == 'dca':
        k = utils.config['k']
        m = compute_m_dca(k, v, d, bit_rate)
        # For each word in vocab (v), for each codebook (m), store log2(k) bits
        # to specify index in codebook this word is using.
        # Must also store the m codebooks, each of which stores k codes,
        # where each code is a d-dimensional full-precision vector.
        mem = v * m * int(np.log2(k)) + m * k * d * 32
    elif utils.config['compresstype'] == 'uniform' and utils.config['skipquant']:
        mem = v * d * 32
    elif utils.config['compresstype'] == 'uniform' and not utils.config['skipquant']:
        # we add 32 because range must be stored
        mem = v * d * bit_rate + 32
    elif utils.config['compresstype'] == 'nocompress':
        mem = v * d * 32
    else:
        raise ValueError('Method name invalid')
    return mem

if __name__ == '__main__':
    main()
