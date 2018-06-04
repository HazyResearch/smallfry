import smallfry as sfry
import numpy as np

path = 'glove.head.txt'
R = 0.8

w2d, sfry_path = sfry.compress(path,"prior.npy",R=R,word_rep ='dict')

inf = np.load(path+'.inflated.npy')

good2go = True
for k in w2d.keys():
    if np.linalg.norm(inf[w2d[k]] - sfry.query(k,w2d,sfry_path)) > 0.01 and len(np.unique(inf[w2d[k]])) > 1:
        print("ERROR: mismatch on word: "+k)
        good2go = False
        break

if good2go:
    print("Inflated npy matches Small-Fry compression queried from file")

smallfry = sfry.load(sfry_path, w2d)
good2go = True

for k in w2d.keys():
    if np.linalg.norm(inf[w2d[k]] - smallfry.query(k)) > 0.01 and len(np.unique(inf[w2d[k]])) > 1:
        print("ERROR: mismatch on word: "+k)
        good2go = False
        break

if good2go:
    print("Inflated npy matches memory mapped Small-Fry compression")



