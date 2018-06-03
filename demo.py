import smallfry as sfry
import numpy as np

path = 'glove.head.txt'
R = 0.8

w2d, sfry_path = sfry.compress(path,"prior.npy",R=R,word_rep ='dict')

inf = np.load(path+'.inflated.npy')

good2go = True
for k in w2d.keys():
    if np.linalg.norm(inf[w2d[k]] - sfry.query2(k,w2d,sfry_path)) > 0.01 and len(np.unique(inf[w2d[k]])) > 1:
        print(k)
        good2go = False
        break

if good2go:
    print("File sfry good 2 go!!!")
else:
    print("OOPS! Bug found -- contact tginart")

smallfry = sfry.load(sfry_path, w2d)

good2go = True

for k in w2d.keys():
    if np.linalg.norm(inf[w2d[k]] - smallfry.query(k)) > 0.01 and len(np.unique(inf[w2d[k]])) > 1:
        print(k)
if good2go:
    print("Mmap sfry good 2 go!!!")
else:
    print("OOPS! Bug found -- contact tginart")



