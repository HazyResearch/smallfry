import smallfry as sfry
import time
import numpy as np

path = 'glove.head.txt'
R = 0.8

tic = time.time()
w2d, sfry_path = sfry.compress(path,"prior.npy",R=R,word_rep ='dict')
toc = time.time()-tic
print(toc)

inf = np.load(path+'.inflated.npy')

good2go = True
for k in w2d.keys():
    if np.linalg.norm(inf[w2d[k]] - sfry.query(k,w2d,sfry_path)) > 0.01 and len(np.unique(inf[w2d[k]])) > 1:
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


#note, once finished, only user will only be responsible for words
#allot_indices and codebks is part of compression -- BUT negligble in size

#r = sfry.query('election',word2idx,dim, sfry_path)
#print("inflated row for 'election' is: "+str(r))

#r = sfry.query('they',word2idx,dim, sfry_path)
#print("inflated row for 'they' is: "+str(r))


#r = sfry.query('.',word2idx,dim, sfry_path)
#print("inflated row for '.' is: "+str(r))
