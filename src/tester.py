import numpy as np
import smallfry as sfry

w2d = np.load('glove.6B.50d.txt.word.npy').item()
dim = 50
inf = np.load('glove.6B.50d.txt.inflated.npy')

for k in w2d.keys():
    if np.linalg.norm(inf[w2d[k]] - sfry.query(k,w2d,'glove.6B.50d.sfry')) > 0.01 and len(np.unique(inf[w2d[k]])) >1:
        print(k)
        break

