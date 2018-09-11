import numpy as np
def codes_2_vec(codes, codebook, m ,k ,v,d):
    codes = codes.reshape(int(len(codes)/m),m)
    dcc_mat = np.zeros([v,d])
    for i in range(v):
        for j in range(m):
            dcc_mat[i,:] += codebook[j*k+codes[i,j],:]
    return dcc_mat