import numpy as np
from scipy.stats import norm
import torch

sigma = 0.1
d = 300
v = 1000
X = 1.0/(np.sqrt(d))*np.random.random([v,d])
w = 1.0/np.sqrt(d)*np.ones([d,1])
y_bar = np.matmul(X,w)
min_val = -1.0/np.sqrt(d)
max_val = 1.0/np.sqrt(d)

def randround(X,b):
    if b == 32:
        return X
    X_torch = torch.Tensor(X)
    scale = (max_val - min_val)/float(2**b -1)
    floor_val = min_val + torch.floor(( X_torch - min_val) / scale) *scale
    ceil_val = min_val + torch.ceil((X_torch - min_val) / scale) * scale
    floor_prob = (ceil_val - X_torch) / scale
    ceil_prob = (X_torch - floor_val) / scale
    sample = torch.FloatTensor(np.random.uniform(size=list(X_torch.size())))
    X_q = floor_val * (sample < floor_prob).float() + ceil_val * (sample >= floor_prob).float()
    return X_q

def compute_gen(X_q, y, sigma, lamb):
    K = np.matmul(X_q,np.transpose(X_q))
    K_sqr = np.matmul(K, np.transpose(K))
    B1 = np.linalg.inv(K + lamb*np.identity(K.shape[0]))
    B1 = np.matmul( np.transpose(B1), B1)
    R = 1.0/v*lamb**2*np.matmul( np.transpose(y) , np.matmul(B1,y))
    R = R + 1.0/v*sigma*np.trace(np.matmul(K_sqr,B1))
    return R

b_s = [1,2,4,8]
lamb_s = [0,0.01,100]
results = np.zeros([len(b_s),len(lamb_s)])

for i in range(len(b_s)):
    b = b_s[i]
    X_q = randround(X,b)
    for j in range(len(lamb_s)):
        l = lamb_s[j]
        results[i,j] = compute_gen(X_q.data.numpy(), y_bar, sigma, l)
