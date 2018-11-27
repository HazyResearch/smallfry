def cmds_11_26_18_compress_round1():
    filename = 'C:\\Users\\avnermay\\Desktop\\Compiled_apps\\11_26_18_compress_round1_cmds'
    prefix = ('qsub -V -b y -wd /proj/mlnlp/avnermay/Babel/wd '
              '/proj/mlnlp/avnermay/Babel/Git/smallfry/src/smallfry_env.sh '
              '\\"python /proj/mlnlp/avnermay/Babel/Git/smallfry/src/compress.py')
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
        bitrates = [1,2,4,8]
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
        # This corresponds to adapt=True, stoch=False, skipquant=True.
        # We choose this combo because skipquant with adapt=False is simply
        # 'nocompress', and when skipquant is True it doesn't matter if stoch
        # is True or False.
        f.write(('{} --rungroup {} --embedtype {} --compresstype {} --bitrate {} '
                '--embeddim {} --seed {} --adaptive --skipquant\\"\n').format(
                prefix, rungroup, embedtype, compresstype, bitrate,
                embeddim, seed)
        )

def cmds_11_26_18_compress_tuneDCA():
    # dca
    filename = 'C:\\Users\\avnermay\\Desktop\\Compiled_apps\\11_26_18_compress_tuneDCA_cmds'
    prefix = ('qsub -V -b y -wd /proj/mlnlp/avnermay/Babel/wd '
              '/proj/mlnlp/avnermay/Babel/Git/smallfry/src/smallfry_env.sh '
              '\\"python /proj/mlnlp/avnermay/Babel/Git/smallfry/src/compress.py')

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

if __name__ == '__main__':
    cmds_11_26_18_compress_round1()
    cmds_11_26_18_compress_tuneDCA()