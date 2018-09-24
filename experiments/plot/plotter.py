import sys
import os
from plot_tools import *
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) #FOR LOCAL IMPORTS

def get_dca_params(results, bitrates, base):
    res = results
    br_2_mks = dict()
    for br in bitrates:
        br_2_mks[br] = []
        for r in res:
            if r == {} or r['base'] != base: continue
            if r['method'] == 'dca' and abs(r['bitrate'] - br) < 0.15*br:
                if 'embed-fro-dist' in r.keys():
                    br_2_mks[br].append((r['m'],r['k'],r['embed-fro-dist']))
                    br_2_mks[br].sort(key=lambda x:x[1])
    return br_2_mks

def get_dca_best_params(results, bitrates, base):
    res = results
    br_2_mk = dict()
    for br in bitrates:
        lowest_mdd = 9999999
        best_res = None
        for r in res:
            if r == {} or r['base'] != base: continue
            if r['method'] == 'dca' and abs(r['bitrate'] - br) < 0.15*br:
                if lowest_mdd > r['embed-fro-dist']:
                    lowest_mdd = r['embed-fro-dist']
                    best_res = r
        br_2_mk[br] = (best_res['m'], best_res['k'])
    return br_2_mk
