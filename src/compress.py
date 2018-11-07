import logging
import math
import time
import numpy as np
import utils
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) #FOR LOCAL IMPORTS

def main():
    utils.init('compress')
    utils.config['base-embed-path'] = get_embed_path()
    base_embeds,wordlist = utils.load_embeddings(utils.config['base-embed-path'])
    (v,d) = base_embeds.shape
    store_embed_info(v,d)
    compressed_embeds = compress_embeddings(base_embeds)
    utils.config['compressed-embed-path'] = utils.get_filename('_compressed_embeds.txt')
    utils.save_embeddings(utils.config['compressed-embed-path'], compressed_embeds, wordlist)
    utils.save_current_config() # override current config

def store_embed_info(v,d):
    assert d == utils.config['embeddim'], 'Embed dims do not match.'
    utils.config['vocab'] = v
    utils.config['memory'] = get_exact_memory()
    utils.config['bitrate'] = utils.config['memory'] / (v * d)
    utils.config['compression-ratio'] = 32 * v * d / utils.config['memory']

def get_embed_path():
    path = ''
    if utils.config['embedtype'] == 'glove400k':
        if utils.config['embeddim'] == 50:
            pass # TODO
        elif utils.config['embeddim'] == 100:
            pass # TODO
        elif utils.config['embeddim'] == 200:
            pass # TODO
        elif utils.config['embeddim'] == 300:
            pass # TODO
    return path

def compress_embeddings(base_embeds):
    start = time.time()
    logging.info('Beggining to make embeddings')
    if utils.config['compresstype'] == 'uniform':
        compressed_embeds = None # TODO
    elif utils.config['compresstype'] == 'kmeans':
        compressed_embeds = None # TODO
    elif utils.config['compresstype'] == 'dca':
        compressed_embeds = None # TODO
    compress_time = time.time() - start
    utils.config['compress-time'] = compress_time
    logging.info('Finished making embeddings. It took {} min.'.format(compress_time/60))
    return compressed_embeds

#     # Save embeddings (text and numpy) and config
#     to_file_txt(core_filename + '.txt', wordlist, embeds)
#     if config['writenpy']:
#         to_file_np(core_filename + '.npy', embeds)
#     save_dict_as_json(config, core_filename + '_config.json')
#     logging.info('maker.py finished!')

# def make_embeddings(base_embeds, embed_dir, config):
#     if config['method'] == 'kmeans':
#         assert config['bitsperblock']/config['blocklen'] == config['ibr'], "intended bitrate for kmeans not met!"
#         start = time.time()
#         sfry = Smallfry.quantize(base_embeds, b=config['bitsperblock'], solver=config['solver'],
#             block_len=config['blocklen'], r_seed=config['seed'])
#         config['sfry-maketime-quantize-secs'] = time.time()-start
#         config['embed-maketime-secs'] = config['sfry-maketime-quantize-secs']
#         start = time.time()
#         embeds = sfry.decode(np.array(list(range(config['vocab']))))
#         config['sfry-maketime-decode-secs'] = time.time()-start
#     elif config['method'] == 'dca':
#         m,k,v,d,ibr = config['m'], config['k'], config['vocab'], config['dim'], config['ibr']
#         #does m make sense given ibr and k?
#         assert m == compute_m_dca(k,v,d,ibr), "m and k does not match intended bit rate"
#         work_dir = str(pathlib.PurePath(embed_dir,'dca_tmp'))
#         start = time.time()
#         compressor = EmbeddingCompressor(m, k, work_dir, 
#                                             config['tau'],
#                                             config['batchsize'],
#                                             config['lr'],
#                                             config['gradclip'])
#         base_embeds = base_embeds.astype(np.float32)
#         dca_train_log = compressor.train(base_embeds)
#         config['dca-maketime-train-secs'] = time.time()-start
#         config['embed-maketime-secs'] = config['dca-maketime-train-secs']
#         me_distance,frob_error = compressor.evaluate(base_embeds)
#         config['mean-euclidean-dist'] = me_distance
#         config['embed-frob-err'] = frob_error
#         with open(work_dir+'.dca-log-json','w+') as log_f:
#             log_f.write(json.dumps(dca_train_log))
#         start = time.time()
#         codes, codebook = compressor.export(base_embeds, work_dir)
#         config['dca-maketime-export-secs'] = time.time() - start
#         start = time.time()
#         codes = np.array(codes).flatten()
#         codebook = np.array(codebook)
#         embeds = codes_2_vec(codes, codebook, m, k, v, d)
#         config['dca-maketime-codes2vec-secs'] = time.time() - start
#     elif config['method'] == 'baseline':
#         assert config['ibr'] == 32.0, "Baselines use floating point precision"
#         embeds = load_embeddings(config['basepath'])[0]
#     elif config['method'] == 'clipnoquant':
#         embeds = load_embeddings(config['basepath'])[0]
#         start = time.time()
#         embeds = clip_no_quant(embeds,config['ibr'])
#         config['embed-maketime-secs'] = time.time()-start
#         config['embed-fro-dist'] = np.linalg.norm(base_embeds - embeds)
#     elif config['method'] == 'midriser':
#         embeds = load_embeddings(config['basepath'])[0]
#         start = time.time()
#         embeds = midriser(base_embeds,config['ibr'])
#         config['embed-maketime-secs'] = time.time()-start
#     elif config['method'] == 'naiveuni':
#         embeds = load_embeddings(config['basepath'])[0]
#         start = time.time()
#         embeds = naiveuni(base_embeds,config['ibr'])
#         config['embed-maketime-secs'] = time.time()-start
#     elif config['method'] == 'stochround':
#         embeds = load_embeddings(config['basepath'])[0]
#         start = time.time()
#         embeds = stochround(base_embeds,config['ibr'])
#         config['embed-maketime-secs'] = time.time()-start
#     elif config['method'] == 'optranuni':
#         embeds = load_embeddings(config['basepath'])[0]
#         start = time.time()
#         embeds = adarange(base_embeds,config['ibr'])
#         config['embed-maketime-secs'] = time.time()-start
#         config['embed-fro-dist'] = np.linalg.norm(base_embeds - embeds)
#     elif config['method'] == 'stochoptranuni':
#         embeds = load_embeddings(config['basepath'])[0]
#         start = time.time()
#         embeds = stoch_adarange(base_embeds,config['ibr'])
#         config['embed-maketime-secs'] = time.time()-start
#         config['embed-fro-dist'] = np.linalg.norm(base_embeds - embeds)
#         #TODO remove this from here
#     elif config['method'] == 'stoch_range_2':
#         embeds = load_embeddings(config['basepath'])[0]
#         start = time.time()
#         embeds = stoch_adarange_2(base_embeds,config['ibr'])
#         config['embed-maketime-secs'] = time.time()-start
#         config['embed-fro-dist'] = np.linalg.norm(base_embeds - embeds)
#         #TODO remove this from here
#     else:
#         raise ValueError(f"Method name invalid {config['method']}")
#     return embeds

# def get_embeddings_dir_and_name(config):
#     if config['method'] == 'kmeans':
#         params = ['base','method','vocab','dim','ibr','bitsperblock','blocklen','seed','date','rungroup','solver']
#     elif config['method'] == 'dca':
#         params = ['base','method','vocab','dim','ibr','m','k','seed','date','rungroup','lr','gradclip','batchsize','tau']
#     elif config['method'] in ['baseline', 'stochround', 'naiveuni', 'midriser','optranuni','clipnoquant','stochoptranuni','stoch_range_2']:
#         params = ['base','method','vocab','dim','ibr','seed','date','rungroup']
#     else:
#         raise ValueError(f"Method name invalid {config['method']}")

#     embed_name_parts = ['{}={}'.format(param, config[param]) for param in params]
#     embed_name = ','.join(embed_name_parts)
#     embed_dir = str(pathlib.PurePath(
#             config['outputdir'],
#             config['date-rungroup'],
#             embed_name
#         ))
#     return embed_dir, embed_name

def get_exact_memory():
    v = utils.config['vocab']
    d = utils.config['dim']
    ibr = utils.config['ibr']
    if utils.config['method'] == 'kmeans':
        num_centroids = 2**ibr
        mem = v * d * ibr + num_centroids * 32
    elif utils.config['method'] == 'dca':
        k = utils.config['k']
        m = utils.compute_m_dca(k, v, d, ibr)
        mem = v * m * int(np.log2(k)) + 32 * d * m * k
    elif utils.config['method'] == 'uniform' and utils.config['skipquant']:
        mem = v * d * 32
    elif utils.config['method'] == 'uniform' and not utils.config['skipquant']:
        # we add 32 because range must be stored
        mem = v * d * ibr + 32
    else:
        raise ValueError('Method name invalid')
    return mem

def compute_m_dca(k, v, d, ibr):
    return int(np.round(ibr * v * d / (v * np.log2(k) + 32 * d * k)))

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

if __name__ == '__main__':
    main()
