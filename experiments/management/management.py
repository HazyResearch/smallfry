import argh
import pathlib
import os
import sys
import glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) #FOR LOCAL IMPORTS
from experimental_utils import * 
'''
MANAEGMENT -- A collection of testbench management routines
'''
def cleanup_rungroup_make(rungroup, basedir=get_base_outputdir()):
    '''
    Cleans a rungroup after making by checking to see if mandatory files are present
    Deletes the embedding directory is not all files are present
    Use with caution
    '''

    def check_file_present(embpath, f_ending):
        check_file_qry = str(pathlib.PurePath(embpath,f"*{f_ending}"))
        print(f"Checking for following files: {check_file_qry}")
        num_files_present = len(glob.glob(check_file_qry))
        assert num_files_present == 1, f"Invalid number of {f_ending} files in embedding: {num_files_present}"

    def delete_emb_dir(embpath):
        print(f"Delete embedding dir: {embpath}")
        os.system(f"rm -rf {embpath}")

    f_endings_to_check = ['maker.log', 'config.json', '.txt']
    qry = str(pathlib.PurePath(basedir,f"{rungroup}/*"))
    for emb in glob.glob(qry):
        for f_ending in f_endings_to_check:
            try:
                check_file_present(emb,f_ending)
            except AssertionError as e:
                print(e)
                delete_emb_dir(emb)


parser = argh.ArghParser()
parser.add_commands([cleanup_rungroup_make])

if __name__ == '__main__':
    parser.dispatch()
