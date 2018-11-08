import logging
import math
import time
import pathlib
import numpy as np
from sklearn.cluster import KMeans
import utils
from third_party.neuralcompressor.nncompress.embed_compress import EmbeddingCompressor

def main():
    utils.init('compress')
    utils.config['base-embed-path'] = get_embed_path()
    base_embeds,wordlist = utils.load_embeddings(utils.config['base-embed-path'])
    (v,d) = base_embeds.shape
    store_embed_info(v,d)
    compressed_embeds = compress_embeddings(base_embeds, utils.config['bitrate'])
    utils.config['compressed-embed-path'] = utils.get_filename('_compressed_embeds.txt')
    utils.save_embeddings(utils.config['compressed-embed-path'], compressed_embeds, wordlist)
    utils.save_current_config() # override current config
    logging.info('Compression complete.  Exiting compress.py main method.')

def store_embed_info(v,d):
    assert d == utils.config['embeddim'], 'Embed dims do not match.'
    utils.config['vocab'] = v
    utils.config['memory'] = get_exact_memory()
    utils.config['bitrate'] = utils.config['memory'] / (v * d)
    utils.config['compression-ratio'] = 32 * v * d / utils.config['memory']

def get_embed_path():
    path = ''
    if utils.config['embedtype'] == 'glove400k':
        path_format_str = str(pathlib.PurePath(utils.config['basedir'],
                              'base_embeddings', 'glove400k', 'glove.6B.{}d.txt'))
        path = path_format_str.format(utils.config['embeddim'])
    return path

def compress_embeddings(X, bit_rate):
    logging.info('Beggining to make embeddings')
    if utils.config['compresstype'] == 'uniform':
        Xq, frob_squared_error, elapsed = compress_uniform(X, bit_rate,
            adaptive_range=False, stochastic_round=False, skip_quantize=False)
    elif utils.config['compresstype'] == 'kmeans':
        Xq, frob_squared_error, elapsed = compress_kmeans(X, bit_rate,
            random_seed=utils.config['seed'], n_init=1)
    elif utils.config['compresstype'] == 'dca':
        Xq, frob_squared_error, elapsed = compress_dca(X, bit_rate,
            utils.config['k'], utils.config['lr'],  utils.config['batchsize'],
            utils.config['temp'], utils.config['gradclip'],
            utils.config['rundir'])
    utils.config['frob-squared-error'] = frob_squared_error
    utils.config['compress-time'] = elapsed
    logging.info('Finished making embeddings. It took {} min.'.format(elapsed/60))
    return Xq

def compress_kmeans(X, bit_rate, random_seed=None, n_init=1):
    # Tony's params for k-means: max_iter=70, n_init=1, tol=0.01.
    # default k-means params: max_iter=300, n_init=10, tol=1e-4
    kmeans = KMeans(n_clusters=2**bit_rate, random_state=random_seed, n_init=n_init)
    start = time.time()
    kmeans = kmeans.fit(X.reshape(X.shape[0] * X.shape[1],1))
    elapsed = time.time() - start
    # map each element of X to the nearest centroid
    Xq = kmeans.cluster_centers_[kmeans.labels_].reshape(X.shape)
    frob_squared_error = kmeans.inertia_
    return Xq, frob_squared_error, elapsed

def compress_dca(X, bit_rate, k, lr, batch_size, temp, grad_clip, rundir):
    # TODO: Test inflate_dca_embeddings
    work_dir = str(pathlib.PurePath(rundir,'dca_tmp'))
    (v,d) = X.shape
    m = compute_m_dca(k, v, d, bit_rate)
    start = time.time()
    compressor = EmbeddingCompressor(m, k, work_dir, temp, batch_size, lr, grad_clip)
    dca_train_log = compressor.train(X)
    elapsed = time.time() - start
    _,frob_squared_error = compressor.evaluate(X)
    codes, codebook = compressor.export(X, work_dir)
    codes = np.array(codes).flatten()
    codebook = np.array(codebook)
    Xq = inflate_dca_embeddings(codes, codebook, m, k, v, d)
    utils.save_dict_as_json(dca_train_log, str(pathlib.PurePath(
        work_dir, utils.config['full-runname'] + '_dca_train_log.json')))
    return Xq, frob_squared_error, elapsed

def compress_uniform(X, bit_rate, adaptive_range=False, stochastic_round=False,
        skip_quantize=False):
    start = time.time()
    if adaptive_range:
        range_limit = find_optimal_range(X, bit_rate)
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
    Xq = np.copy(X)
    if get_max_abs(Xq) > range_limit:
        np.clip(Xq, -range_limit, range_limit, out=Xq)
    if bit_rate < 32 and range_limit != 0 and not skip_quantize:
        # affine transform to put Xq in [0,2**bit_rate - 1]
        Xq = (2**bit_rate - 1) * (Xq + range_limit) / (2 * range_limit)
        if stochastic_round:
            # each entry will round down if noise > fraction part
            np.ceil(Xq - np.random.rand(*Xq.shape), out=Xq)
        else:
            np.round(Xq, out=Xq)
        # undo affine transformation
        Xq = (Xq * 2 * range_limit) / (2**bit_rate - 1) - range_limit
    return Xq

def find_optimal_range(X, bit_rate, stochastic_round=False, tol=1e-2):
    '''
    Find the best value to use to clip the embeddings before using uniform quantization.
    '''
    f = lambda range_limit : quantize_and_compute_frob_error(
        X, bit_rate, range_limit, stochastic_round=stochastic_round)

    return golden_section_search(f, 0, get_max_abs(X), tol=tol)

def quantize_and_compute_frob_error(X, bit_rate, range_limit, stochastic_round=False):
    '''
    Function which computes Frob error after quantizing (ADD MORE DESCRIPTION).
    '''
    X_q = _compress_uniform(X, bit_rate, range_limit, stochastic_round=stochastic_round)
    return np.linalg.norm(X - X_q) # frob norm

def golden_section_search(f, x_min=1e-10, x_max=10, tol=1e-2):
    '''
    Find argmin of f between x_min and x_max (for f uni-modal).
    
    This function uses the golden-section search algorithm.
    It always maintains a list of four points [x1,x2,x3,x4],
    which are always spaced as: [a,a+(c^2)h,a+ch,a+h].
    for c = (math.sqrt(5) - 1) / 2 = 0.618...
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
    #initialize points
    phi = (math.sqrt(5) - 1) / 2
    x1 = x_min
    x4 = x_max
    f_x1 = f(x1)
    f_x4 = f(x4)
    x2 = x1 + (x4-x1) * phi**2
    x3 = x1 + (x4-x1) * phi
    f_x2 = f(x2)
    f_x3 = f(x3)
    while (x4-x1 > tol):
        assert (math.isclose(x2, x1 + (x4 - x1) * phi**2) and 
                math.isclose(x3, x1 + (x4 - x1) * phi))
        if f_x2 < f_x3:
            # The new points become [x1, NEW, x2, x3]
            x4,f_x4 = x3,f_x3
            x3,f_x3 = x2,f_x2
            x2 = x1 + (x4-x1) * phi**2
            f_x2 = f(x2)
        else:
            # The new points become [x2, x3, NEW, x4]
            x1,f_x1 = x2,f_x2
            x2,f_x2 = x3,f_x3
            x3 = x1 + (x4-x1) * phi
            f_x3 = f(x3)
        
    # Return x-value with minimum f(x) which was found.
    i = np.argmin([f_x1,f_x2,f_x3,f_x4])
    x = [x1,x2,x3,x4]
    return x[i]

def get_max_abs(X):
    return np.max(np.abs(X))

def inflate_dca_embeddings(codes, codebook, m, k , v, d):
    ''' reshapes inflates DCA output embeddings -- assumes input is properly formatted and flattened'''
    codes = codes.reshape(int(len(codes)/m),m)
    embeds = np.zeros([v,d])
    for i in range(v):
        for j in range(m):
            embeds[i,:] += codebook[j*k+codes[i,j],:]
    return embeds

def compute_m_dca(k, v, d, bit_rate):
    return int(np.round(bit_rate * v * d / (v * np.log2(k) + 32 * d * k)))

def get_exact_memory():
    v = utils.config['vocab']
    d = utils.config['dim']
    bit_rate = utils.config['bitrate']
    if utils.config['method'] == 'kmeans':
        num_centroids = 2**bit_rate
        mem = v * d * bit_rate + num_centroids * 32
    elif utils.config['method'] == 'dca':
        k = utils.config['k']
        m = compute_m_dca(k, v, d, bit_rate)
        mem = v * m * int(np.log2(k)) + 32 * d * m * k
    elif utils.config['method'] == 'uniform' and utils.config['skipquant']:
        mem = v * d * 32
    elif utils.config['method'] == 'uniform' and not utils.config['skipquant']:
        # we add 32 because range must be stored
        mem = v * d * bit_rate + 32
    else:
        raise ValueError('Method name invalid')
    return mem

if __name__ == '__main__':
    main()
