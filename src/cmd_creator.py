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
    with open(filename,'w+') as f:
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

    with open(filename,'w+') as f:
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
    with open(filename,'w+') as f:
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
    with open(cmd_file,'w+') as f:
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
    with open(cmd_file,'w+') as f:
        for path in embed_file_paths:
            f.write(cmd_format_str.format(path.strip()))

if __name__ == '__main__':
    # cmds_11_28_18_compress_round1()
    # cmds_11_28_18_compress_tuneDCA()
    # cmds_11_28_18_compress_fiveSeeds()
    # cmds_11_29_18_eval_fiveSeeds()
    cmds_11_29_18_eval_fiveSeeds_fixFailedEvals()
