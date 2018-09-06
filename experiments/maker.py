import smallfry
import pathlib
import numpy as np
import os
import argparse
import datetime

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
    parser.add_argument('--bitrate', type=int,
        help='Bits per block')
    parser.add_argument('--blocklen', type=int, default=1,
        help='Block length for quantization/k-means')

    return parser


def make_kmeans(base, base_path, bitrate, block_len, r_seed, output_dir):
    embeds, wordlist = smallfry.utils.load_embeddings(base_path)
    sfry = smallfry.smallfry.Smallfry.quantize(embeds,b=bitrate,block_len=block_len,r_seed=r_seed)
    sfry_embeds = sfry.decode(np.array(list(range(len(wordlist)))))
    v = sfry_embeds.shape[0]
    d = sfry_embeds.shape[1]
    mem = v*d*bitrate + 2**bitrate*32*block_len
    embed_name, embed_dir = create_filename_kmeans(output_dir, base, bitrate, block_len, r_seed, v, d, mem)
    os.makedirs(embed_dir)
    to_file_txt(pathlib.PurePath(embed_dir, embed_name + '.txt'), wordlist, sfry_embeds)
    to_file_np(pathlib.PurePath(embed_dir, embed_name), sfry_embeds)


def create_filename_kmeans(output_folder, base, bitrate, block_len, r_seed, v, d, mem):
    embed_name = "base=" + str(base)\
				+",vocab=" + str(v)\
                +",dim=" + str(d)\
				+ ",bitrate=" + str(bitrate)\
				+ ",blocklen=" + str(block_len)\
				+ ",rseed=" + str(r_seed)\
				+ ",mem=" + str(mem)\
				+ ",datetime=" + get_date_str()
    return embed_name, pathlib.PurePath(output_folder, "seed="+str(r_seed), embed_name)


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

def main():
    args = vars(init_parser().parse_args())
    make_kmeans(args['base'], args['basepath'], args['bitrate'], args['blocklen'], args['seed'], args['output_dir'])



if __name__ == '__main__':
    main()
