import glob
import utils
import pathlib

def get_cmdfile_path(filename):
    return str(pathlib.PurePath(utils.get_base_dir(), 'scripts', filename))

def cmds_11_28_18_compress_round1():
    filename = get_cmdfile_path('11_28_18_compress_round1_cmds')
    prefix = ('qsub -V -b y -wd /proj/smallfry/wd '
              '/proj/smallfry/git/smallfry/src/smallfry_env.sh '
              '\\"python /proj/smallfry/git/smallfry/src/compress.py')
    rungroup = 'round1'
    embedtype = 'glove400k'
    seed = 1
    with open(filename,'w') as f:
        # nocompress
        compresstype = 'nocompress'
        bitrate = 32
        embeddims = [50,100,200,300]
        for embeddim in embeddims:
            f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                    '--embeddim {} --seed {}\\"\n').format(
                prefix, rungroup, embedtype, compresstype, bitrate, embeddim, seed)
            )

        # bitrates and embeddim for k-means and uniform
        bitrates = [1,2,4] # kmeans failed on bitrate 8
        embeddim = 300

        # kmeans
        compresstype = 'kmeans'
        for bitrate in bitrates:
            f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                    '--embeddim {} --seed {}\\"\n').format(
                prefix, rungroup, embedtype, compresstype, bitrate, embeddim, seed)
            )

        # uniform
        compresstype = 'uniform'
        adapts = [False,True]
        stochs = [False,True]
        #skipquant = False
        for bitrate in bitrates:
            for adapt in adapts:
                for stoch in stochs:
                    adapt_str = ' --adaptive' if adapt else ''
                    stoch_str = ' --stoch' if stoch else ''
                    f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                            '--embeddim {} --seed {}{}{}\\"\n').format(
                            prefix, rungroup, embedtype, compresstype, bitrate,
                            embeddim, seed, adapt_str, stoch_str)
                    )
            # ** Ablation for uniform quantization: clipping without quantizing. **
            # For each bitrate, we only consider adapt=True, stoch=False, skipquant=True.
            # We choose this combo because skipquant with adapt=False is simply
            # 'nocompress', and when skipquant is True it doesn't matter if stoch
            # is True or False.
            f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                '--embeddim {} --seed {} --adaptive --skipquant\\"\n').format(
                prefix, rungroup, embedtype, compresstype, bitrate,
                embeddim, seed)
            )

def cmds_11_28_18_compress_tuneDCA():
    # dca
    filename = get_cmdfile_path('11_28_18_compress_tuneDCA_cmds')
    prefix = ('qsub -V -b y -wd /proj/smallfry/wd '
              '/proj/smallfry/git/smallfry/src/dca_docker.sh '
              '\\"python /proj/smallfry/git/smallfry/src/compress.py')
    rungroup = 'tuneDCA'
    compresstype = 'dca'
    embeddim = 300
    embedtype = 'glove400k'
    seed = 1

    # 60 total configurations
    bitrates = [1,2,4] # 3
    ks = [2,4,8,16] # 4
    lrs = ['0.00001', '0.00003', '0.0001', '0.0003', '0.001'] # 5

    with open(filename,'w') as f:
        for bitrate in bitrates:
            for k in ks:
                for lr in lrs:
                    f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                        '--embeddim {} --seed {} --k {} --lr {}\\"\n').format(
                        prefix, rungroup, embedtype, compresstype, bitrate,
                        embeddim, seed, k, lr)
                    )

def cmds_11_28_18_compress_fiveSeeds():
    filename = get_cmdfile_path('11_28_18_compress_fiveSeeds_cmds')
    prefix = ('qsub -V -b y -wd /proj/smallfry/wd '
              '/proj/smallfry/git/smallfry/src/smallfry_env.sh '
              '\\"python /proj/smallfry/git/smallfry/src/compress.py')
    dca_prefix = ('qsub -V -b y -wd /proj/smallfry/wd '
              '/proj/smallfry/git/smallfry/src/dca_docker.sh '
              '\\"python /proj/smallfry/git/smallfry/src/compress.py')
    rungroup = 'fiveSeeds'
    embedtype = 'glove400k'
    seeds = [1,2,3,4,5]
    with open(filename,'w') as f:
        for seed in seeds:
            # nocompress
            compresstype = 'nocompress'
            bitrate = 32
            embeddims = [50,100,200,300]
            for embeddim in embeddims:
                f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                        '--embeddim {} --seed {}\\"\n').format(
                    prefix, rungroup, embedtype, compresstype, bitrate, embeddim, seed)
                )

            # bitrates and embeddim for k-means, uniform, and dca
            bitrates = [1,2,4] # kmeans failed on bitrate 8
            embeddim = 300

            # kmeans
            compresstype = 'kmeans'
            for bitrate in bitrates:
                f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                        '--embeddim {} --seed {}\\"\n').format(
                    prefix, rungroup, embedtype, compresstype, bitrate, embeddim, seed)
                )

            # uniform
            compresstype = 'uniform'
            adapts = [False,True]
            stochs = [False,True]
            #skipquant = False
            for bitrate in bitrates:
                for adapt in adapts:
                    for stoch in stochs:
                        adapt_str = ' --adaptive' if adapt else ''
                        stoch_str = ' --stoch' if stoch else ''
                        f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                                '--embeddim {} --seed {}{}{}\\"\n').format(
                                prefix, rungroup, embedtype, compresstype, bitrate,
                                embeddim, seed, adapt_str, stoch_str)
                        )
                # ** Ablation for uniform quantization: clipping without quantizing. **
                # For each bitrate, we only consider adapt=True, stoch=False, skipquant=True.
                # We choose this combo because skipquant with adapt=False is simply
                # 'nocompress', and when skipquant is True it doesn't matter if stoch
                # is True or False.
                f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                    '--embeddim {} --seed {} --adaptive --skipquant\\"\n').format(
                    prefix, rungroup, embedtype, compresstype, bitrate,
                    embeddim, seed)
                )

            # dca
            compresstype = 'dca'
            # These are the best bitrate,k,lr combos from the 2018-11-28-tuneDCA run (keys are 'b' value).
            # I ran 'dca_get_best_k_lr_per_bitrate()' in plotter.py to compute the above
            # dictionary containing the best performing settings.
            ### import plotter
            ### bitrate_k_lr = plotter.dca_get_best_k_lr_per_bitrate()
            bitrate_k_lr = {1: {'k': 4, 'lr': 0.0003},
                            2: {'k': 4, 'lr': 0.0003},
                            4: {'k': 8, 'lr': 0.0003}}
            for bitrate in bitrates:
                k = bitrate_k_lr[bitrate]['k']
                lr = bitrate_k_lr[bitrate]['lr']
                f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                        '--embeddim {} --seed {} --k {} --lr {}\\"\n').format(
                        dca_prefix, rungroup, embedtype, compresstype, bitrate,
                        embeddim, seed, k, lr)
                )

def cmds_11_29_18_eval_fiveSeeds():
    cmd_file = get_cmdfile_path('11_29_18_eval_fiveSeeds_cmds')
    if utils.is_windows():
        embed_list_file = get_cmdfile_path('embedding_list_fiveSeeds.txt')
    else:
        embed_list_file = '/proj/smallfry/embeddings/glove400k/2018-11-29-fiveSeeds/embedding_list_fiveSeeds.txt'
    cmd_format_str = ('qsub -V -b y -wd /proj/smallfry/wd /proj/smallfry/git/smallfry/src/smallfry_env.sh '
              '\\"python /proj/smallfry/git/smallfry/src/evaluate.py --cuda --evaltype qa --embedpath {}\\"\n')

    with open(embed_list_file,'r') as f:
        embed_file_paths = f.readlines()
    with open(cmd_file,'w') as f:
        for path in embed_file_paths:
            f.write(cmd_format_str.format(path.strip()))

def cmds_11_29_18_eval_fiveSeeds_fixFailedEvals():
    cmd_file = get_cmdfile_path('11_29_18_eval_fiveSeeds_fixFailedEvals_cmds')
    embed_file_paths = [
        '/proj/smallfry/embeddings/glove400k/2018-11-29-fiveSeeds/seed,3_embeddim,300_compresstype,uniform_bitrate,4_stoch,True_adaptive,True/embedtype,glove400k_rungroup,2018-11-29-fiveSeeds_seed,3_embeddim,300_compresstype,uniform_bitrate,4_stoch,True_adaptive,True_compressed_embeds.txt',
        '/proj/smallfry/embeddings/glove400k/2018-11-29-fiveSeeds/seed,4_embeddim,300_compresstype,uniform_bitrate,2_adaptive,True/embedtype,glove400k_rungroup,2018-11-29-fiveSeeds_seed,4_embeddim,300_compresstype,uniform_bitrate,2_adaptive,True_compressed_embeds.txt'
    ]
    cmd_format_str = ('qsub -V -b y -wd /proj/smallfry/wd /proj/smallfry/git/smallfry/src/smallfry_env.sh '
              '\\"python /proj/smallfry/git/smallfry/src/evaluate.py --cuda --evaltype qa --embedpath {}\\"\n')
    with open(cmd_file,'w') as f:
        for path in embed_file_paths:
            f.write(cmd_format_str.format(path.strip()))

def cmds_12_14_18_trainGlove():
    filename = get_cmdfile_path('12_14_18_trainGlove_cmds')
    cmd_format_str = ('qsub -V -b y -wd /proj/smallfry/wd '
              '/proj/smallfry/git/smallfry/src/smallfry_env.sh '
              '\\"python /proj/smallfry/git/smallfry/src/train_glove.py '
              '--embedtype glove --corpus wiki --rungroup {} --embeddim {} --threads 72\\"\n')
    rungroup = 'trainGlove'
    dims = [25,50,100,200,400,800,1600] # note: 1600 failed to run.
    with open(filename,'w') as f:
        for dim in dims:
            f.write(cmd_format_str.format(rungroup, dim))

def cmds_12_15_18_fasttext_tuneDCA():
    # dca
    filename = get_cmdfile_path('12_15_18_fasttext_tuneDCA')
    prefix = ('qsub -V -b y -wd /proj/smallfry/wd '
              '/proj/smallfry/git/smallfry/src/dca_docker.sh '
              '\\"python /proj/smallfry/git/smallfry/src/compress.py')
    rungroup = 'fasttextTuneDCA'
    compresstype = 'dca'
    embeddim = 300
    embedtype = 'fasttext1m'
    seed = 1

    # 60 total configurations
    bitrates = [1,2,4] # 3
    ks = [2,4,8,16] # 4
    lrs = ['0.00001', '0.00003', '0.0001', '0.0003', '0.001'] # 5

    with open(filename,'w') as f:
        for bitrate in bitrates:
            for k in ks:
                for lr in lrs:
                    f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                        '--embeddim {} --seed {} --k {} --lr {}\\"\n').format(
                        prefix, rungroup, embedtype, compresstype, bitrate,
                        embeddim, seed, k, lr)
                    )

def cmds_12_15_18_compress_dimVsPrec():
    filename = get_cmdfile_path('12_15_18_compress_dimVsPrec')
    prefix = ('qsub -V -b y -wd /proj/smallfry/wd '
              '/proj/smallfry/git/smallfry/src/smallfry_env.sh '
              '\\"python /proj/smallfry/git/smallfry/src/compress.py')
    rungroup = 'dimVsPrec'
    embedtype = 'glove-wiki-am'
    seeds = [1,2,3,4,5]
    embeddims = [25,50,100,200,400,800]
    with open(filename,'w') as f:
        for seed in seeds:
            for embeddim in embeddims:
                # nocompress
                compresstype = 'nocompress'
                bitrate = 32
                f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                        '--embeddim {} --seed {}\\"\n').format(
                    prefix, rungroup, embedtype, compresstype, bitrate, embeddim, seed)
                )

                # bitrates for uniform quantization
                bitrates = [1,2,4,8,16] # kmeans failed on bitrate 8

                # uniform
                compresstype = 'uniform'
                adapt = True
                stochs = [False,True]
                #skipquant = False
                for bitrate in bitrates:
                    for stoch in stochs:
                        adapt_str = ' --adaptive' if adapt else ''
                        stoch_str = ' --stoch' if stoch else ''
                        f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                                '--embeddim {} --seed {}{}{}\\"\n').format(
                                prefix, rungroup, embedtype, compresstype, bitrate,
                                embeddim, seed, adapt_str, stoch_str)
                        )

def cmds_wiki_400k_create_cooccur():
    filename = get_cmdfile_path('12_17_18_create_cooccur_wiki400k')
    prefix = ('qsub -V -b y -wd /proj/smallfry/wd '
              '/proj/smallfry/git/smallfry/src/create_cooccur.sh ')
    BUILDDIR = '/proj/smallfry/git/smallfry/src/third_party/GloVe/build'
    CORPUS = '/proj/smallfry/corpora/wiki/wiki.en.txt'
    VOCAB_FILE = '/proj/smallfry/corpora/wiki/vocab_wiki400k.txt'
    COOCCURRENCE_FILE = '/proj/smallfry/corpora/wiki/cooccurrence_wiki400k.bin'
    COOCCURRENCE_SHUF_FILE = '/proj/smallfry/corpora/wiki/cooccurrence_wiki400k.shuf.bin'
    MAX_VOCAB = 400000
    MEMORY = 160
    with open(filename,'w') as f:
        f.write('{} {} {} {} {} {} {} {}\n'.format(
            prefix, BUILDDIR, CORPUS, VOCAB_FILE, COOCCURRENCE_FILE,
            COOCCURRENCE_SHUF_FILE, MAX_VOCAB, MEMORY
        ))

def cmds_12_17_18_trainGlove_wiki400k():
    filename = get_cmdfile_path('12_17_18_trainGlove_wiki400k_cmds')
    cmd_format_str = ('qsub -V -b y -wd /proj/smallfry/wd '
              '/proj/smallfry/git/smallfry/src/smallfry_env.sh '
              '\\"python /proj/smallfry/git/smallfry/src/train_glove.py '
              '--embedtype glove --corpus {} --rungroup {} --embeddim {} --threads 72\\"\n')
    corpus = 'wiki400k'
    rungroup = 'trainGlove'
    dims = [25,50,100,200,400,800,1600] # note: 1600 failed to run.
    with open(filename,'w') as f:
        for dim in dims:
            f.write(cmd_format_str.format(corpus, rungroup, dim))

def cmds_12_18_18_trainGlove_wiki400k():
    filename = get_cmdfile_path('12_18_18_trainGlove_wiki400k_cmds')
    cmd_format_str = ('qsub -V -b y -wd /proj/smallfry/wd '
              '/proj/smallfry/git/smallfry/src/smallfry_env.sh '
              '\\"python /proj/smallfry/git/smallfry/src/train_glove.py '
              '--embedtype glove --corpus {} --rungroup {} --embeddim {} --lr {} --threads 72\\"\n')
    corpus = 'wiki400k'
    rungroup = 'trainGlove'
    dim = 800 # note: 1600 failed to run.
    lrs = [0.025, 0.01]
    with open(filename,'w') as f:
        for lr in lrs:
            f.write(cmd_format_str.format(corpus, rungroup, dim, lr))

def cmds_12_18_18_compress_fastText_FiveSeeds():
    filename = get_cmdfile_path('12_18_18_compress_fastText_FiveSeeds_cmds')
    prefix = ('qsub -V -b y -wd /proj/smallfry/wd '
              '/proj/smallfry/git/smallfry/src/smallfry_env.sh '
              '\\"python /proj/smallfry/git/smallfry/src/compress.py')
    dca_prefix = ('qsub -V -b y -wd /proj/smallfry/wd '
              '/proj/smallfry/git/smallfry/src/dca_docker.sh '
              '\\"python /proj/smallfry/git/smallfry/src/compress.py')
    rungroup = 'fiveSeeds'
    embedtype = 'fasttext1m'
    seeds = [1,2,3,4,5]
    with open(filename,'w') as f:
        for seed in seeds:
            # nocompress
            compresstype = 'nocompress'
            bitrate = 32
            embeddim = 300
            f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                    '--embeddim {} --seed {}\\"\n').format(
                prefix, rungroup, embedtype, compresstype, bitrate, embeddim, seed)
            )

            # bitrates and embeddim for k-means, uniform, and dca
            bitrates = [1,2,4] # kmeans failed on bitrate 8
            embeddim = 300

            # kmeans
            compresstype = 'kmeans'
            for bitrate in bitrates:
                f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                        '--embeddim {} --seed {}\\"\n').format(
                    prefix, rungroup, embedtype, compresstype, bitrate, embeddim, seed)
                )

            # uniform
            compresstype = 'uniform'
            adapts = [False,True]
            stochs = [False,True]
            #skipquant = False
            for bitrate in bitrates:
                for adapt in adapts:
                    for stoch in stochs:
                        adapt_str = ' --adaptive' if adapt else ''
                        stoch_str = ' --stoch' if stoch else ''
                        f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                                '--embeddim {} --seed {}{}{}\\"\n').format(
                                prefix, rungroup, embedtype, compresstype, bitrate,
                                embeddim, seed, adapt_str, stoch_str)
                        )
                # ** Ablation for uniform quantization: clipping without quantizing. **
                # For each bitrate, we only consider adapt=True, stoch=False, skipquant=True.
                # We choose this combo because skipquant with adapt=False is simply
                # 'nocompress', and when skipquant is True it doesn't matter if stoch
                # is True or False.
                f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                    '--embeddim {} --seed {} --adaptive --skipquant\\"\n').format(
                    prefix, rungroup, embedtype, compresstype, bitrate,
                    embeddim, seed)
                )

            # dca
            compresstype = 'dca'
            # These are the best bitrate,k,lr combos from the 2018-12-16-fasttextTuneDCA run (keys are 'b' value).
            # I ran 'dca_get_best_k_lr_per_bitrate(regex)' in plotter.py to compute the
            # dictionary below containing the best performing settings.
            ### path_regex = '/proj/smallfry/embeddings/fasttext1m/2018-12-16-fasttextTuneDCA/*/*final.json'
            ### best = plotter.dca_get_best_k_lr_per_bitrate(path_regex)
            bitrate_k_lr = {1: {'k': 8, 'lr': 0.0001},
                            2: {'k': 4, 'lr': 0.0001},
                            4: {'k': 8, 'lr': 0.0001}}
            for bitrate in bitrates:
                k = bitrate_k_lr[bitrate]['k']
                lr = bitrate_k_lr[bitrate]['lr']
                f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                        '--embeddim {} --seed {} --k {} --lr {}\\"\n').format(
                        dca_prefix, rungroup, embedtype, compresstype, bitrate,
                        embeddim, seed, k, lr)
                )

def cmds_12_18_18_compress_fastText_FiveSeeds_dca():
    filename = get_cmdfile_path('12_18_18_compress_fastText_FiveSeeds_dca_cmds')
    dca_prefix = ('qsub -V -b y -wd /proj/smallfry/wd '
              '/proj/smallfry/git/smallfry/src/dca_docker.sh '
              '\\"python /proj/smallfry/git/smallfry/src/compress.py')
    rungroup = 'fiveSeeds'
    embedtype = 'fasttext1m'
    compresstype = 'dca'
    seeds = [1,2,3,4,5]
    bitrates = [1,2,4] # kmeans failed on bitrate 8
    embeddim = 300
    # These are the best bitrate,k,lr combos from the 2018-12-16-fasttextTuneDCA run (keys are 'b' value).
    # I ran 'dca_get_best_k_lr_per_bitrate(regex)' in plotter.py to compute the
    # dictionary below containing the best performing settings.
    ### path_regex = '/proj/smallfry/embeddings/fasttext1m/2018-12-16-fasttextTuneDCA/*/*final.json'
    ### best = plotter.dca_get_best_k_lr_per_bitrate(path_regex)
    bitrate_k_lr = {1: {'k': 8, 'lr': 0.0001},
                    2: {'k': 4, 'lr': 0.0001},
                    4: {'k': 8, 'lr': 0.0001}}
    with open(filename,'w') as f:
        for seed in seeds:
            for bitrate in bitrates:
                k = bitrate_k_lr[bitrate]['k']
                lr = bitrate_k_lr[bitrate]['lr']
                f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                        '--embeddim {} --seed {} --k {} --lr {}\\"\n').format(
                        dca_prefix, rungroup, embedtype, compresstype, bitrate,
                        embeddim, seed, k, lr)
                )


def cmds_12_19_18_compress_gloveWiki400k_dimVsPrec():
    filename = get_cmdfile_path('12_19_18_compress_gloveWiki400k_dimVsPrec')
    prefix = ('qsub -V -b y -wd /proj/smallfry/wd '
              '/proj/smallfry/git/smallfry/src/smallfry_env.sh '
              '\\"python /proj/smallfry/git/smallfry/src/compress.py')
    rungroup = 'dimVsPrec'
    embedtype = 'glove-wiki400k-am'
    seeds = [1,2,3,4,5]
    embeddims = [25,50,100,200,400,800]
    with open(filename,'w') as f:
        for seed in seeds:
            for embeddim in embeddims:
                # nocompress
                compresstype = 'nocompress'
                bitrate = 32
                f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                        '--embeddim {} --seed {}\\"\n').format(
                    prefix, rungroup, embedtype, compresstype, bitrate, embeddim, seed)
                )

                # bitrates for uniform quantization
                bitrates = [1,2,4,8,16] # kmeans failed on bitrate 8

                # uniform
                compresstype = 'uniform'
                adapt = True
                stochs = [False,True]
                #skipquant = False
                for bitrate in bitrates:
                    for stoch in stochs:
                        adapt_str = ' --adaptive' if adapt else ''
                        stoch_str = ' --stoch' if stoch else ''
                        f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                                '--embeddim {} --seed {}{}{}\\"\n').format(
                                prefix, rungroup, embedtype, compresstype, bitrate,
                                embeddim, seed, adapt_str, stoch_str)
                        )

def cmds_12_19_18_eval_drqa_fasttext_gloveWiki400k():
    cmd_files = [get_cmdfile_path('12_19_18_eval_drqa_fasttext_fiveSeeds_cmds'),
                 get_cmdfile_path('12_19_18_eval_drqa_gloveWiki400k_dimVsPrec_cmds')]
    path_regexes = ['/proj/smallfry/embeddings/fasttext1m/2018-12-19-fiveSeeds/*/*embeds.txt',
                    '/proj/smallfry/embeddings/glove-wiki400k-am/2018-12-19-dimVsPrec/*/*embeds.txt']
    cmd_format_str = ('qsub -V -b y -wd /proj/smallfry/wd /proj/smallfry/git/smallfry/src/smallfry_env.sh '
              '\\"python /proj/smallfry/git/smallfry/src/evaluate.py --cuda --evaltype qa --embedpath {}\\"\n')
    for i in range(len(cmd_files)):
        cmd_file = cmd_files[i]
        path_regex = path_regexes[i]
        embed_file_paths = glob.glob(path_regex)
        with open(cmd_file,'w') as f:
            for path in embed_file_paths:
                f.write(cmd_format_str.format(path.strip()))

if __name__ == '__main__':
    # cmds_11_28_18_compress_round1()
    # cmds_11_28_18_compress_tuneDCA()
    # cmds_11_28_18_compress_fiveSeeds()
    # cmds_11_29_18_eval_fiveSeeds()
    # cmds_11_29_18_eval_fiveSeeds_fixFailedEvals()
    # cmds_12_14_18_trainGlove()
    # cmds_12_15_18_fasttext_tuneDCA()
    # cmds_12_15_18_compress_dimVsPrec()
    # cmds_wiki_400k_create_cooccur()
    # cmds_12_17_18_trainGlove_wiki400k()
    # cmds_12_18_18_trainGlove_wiki400k()
    # cmds_12_18_18_compress_fastText_FiveSeeds()
    # cmds_12_18_18_compress_fastText_FiveSeeds_dca()
    # cmds_12_19_18_compress_gloveWiki400k_dimVsPrec()
    cmds_12_19_18_eval_drqa_fasttext_gloveWiki400k()
