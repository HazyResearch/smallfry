import smallfry
import pathlib
import numpy as np
import os
import argparse
import datetime
from neuralcompressor.nncompress import EmbeddingCompressor
from subprocess import check_output

def init_parser():
    """Initialize Cmd-line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=True,
        help='Name of base embeddings')
    parser.add_argument('--basepath', type=str, required=True,
        help='Path to base embeddings')
    parser.add_argument('--seed', type=int, required=True,
        help='Random seed to use for experiment.')
    parser.add_argument('--output_dir', type=str, required=True,
        help='Head output directory')
    parser.add_argument('--rungroup`', type=str, required=True,
        help='Rungroup for organization')
    parser.add_argument('--bitrate', type=int,
        help='Bits per block')
    parser.add_argument('--blocklen', type=int, default=1,
        help='Block length for quantization/k-means')

    return parser


def make_dca(base, base_path, M, K, r_seed, output_dir):
    embeds, wordlist = smallfry.utils.load_embeddings(base_path)
    work = pathlib.PurePath
    compressor = EmbeddingCompressor(M, K, work_dir)
    compressor.train(embeds)
    codes, codebook = compressor.export
    inflated_embeds = codes_2_vec(codes, codebook)
    mem = v*M*np.log2(K) + 32*d*M*K
    embed_name, embed_dir = create_filename_dca(output_dir, base, M, K, r_seed, v, d, mem, rungroup)
    os.makedirs(embed_dir)
    to_file_txt(pathlib.PurePath(embed_dir, embed_name + '.txt'), wordlist, inflated_embeds)
    to_file_np(pathlib.PurePath(embed_dir, embed_name), inflated_embeds)


def codes_2_vec(codes, codebook):


def make_kmeans(base, base_path, bitrate, block_len, r_seed, output_dir):
    embeds, wordlist = smallfry.utils.load_embeddings(base_path)
    sfry = smallfry.smallfry.Smallfry.quantize(embeds,b=bitrate,block_len=block_len,r_seed=r_seed)
    sfry_embeds = sfry.decode(np.array(list(range(len(wordlist)))))
    v = sfry_embeds.shape[0]
    d = sfry_embeds.shape[1]
    mem = v*d*bitrate + 2**bitrate*32*block_len
    embed_name, embed_dir = create_filename_kmeans(output_dir, base, bitrate, block_len, r_seed, v, d, mem, rungroup)
    os.makedirs(embed_dir)
    to_file_txt(pathlib.PurePath(embed_dir, embed_name + '.txt'), wordlist, sfry_embeds)
    to_file_np(pathlib.PurePath(embed_dir, embed_name), sfry_embeds)
    return embed_dir

def create_filename_dca(output_folder, base, M , K, r_seed, v, d, mem, rungroup):
    embed_name = "base=" + str(base)\
				+",vocab=" + str(v)\
                +",dim=" + str(d)\
				+ ",M=" + str(M)\
				+ ",K=" + str(K)\
				+ ",rseed=" + str(r_seed)\
				+ ",mem=" + str(mem)\
				+ ",datetime=" + get_date_str()
    return embed_name, pathlib.PurePath(output_folder, get_date_str()+"rungroup="+rungroup, embed_name)

def create_filename_kmeans(output_folder, base, bitrate, block_len, r_seed, v, d, mem, rungroup):
    embed_name = "base=" + str(base)\
				+",vocab=" + str(v)\
                +",dim=" + str(d)\
				+ ",bitrate=" + str(bitrate)\
				+ ",blocklen=" + str(block_len)\
				+ ",rseed=" + str(r_seed)\
				+ ",mem=" + str(mem)\
				+ ",datetime=" + get_date_str()
    return embed_name, pathlib.PurePath(output_folder, get_date_str()+"rungroup="+rungroup, embed_name)


def get_date_str():
	return '{:%Y-%m-%d}'.format(datetime.date.today())

def to_file_np(path, embeds):
    np.save(path, embeds)

def to_file_txt(path, wordlist, embeds):
    with open(path, "w+") as file:
        for i in range(len(wordlist)):
            file.write(wordlist[i] + " ")
            row = embeds[i, :]
            strrow = [str(r) for r in row]
            file.write(" ".join(strrow))
            file.write("\n")

def get_git_hash():
   git_hash = None
   try:
       git_hash = check_output(['git','rev-parse','--short','HEAD']).strip()
       logging.info('Git hash {}'.format(git_hash))
   except FileNotFoundError:
       logging.info('Unable to get git hash.')
   return git_hash


def main():
    args = vars(init_parser().parse_args())
    embed_dir = make_kmeans(args['base'], args['basepath'], args['bitrate'], args['blocklen'], args['seed'], args['output_dir'])
    args['githash'] = get_git_hash() 
    with open(pathlib.PurePath(embed_dir, 'creation_args.json'), 'w+') as f:
        f.write(json.dumps(args))


if __name__ == '__main__':
    main()
