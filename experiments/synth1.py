import numpy as np
from scipy.stats import norm
import torch

sigma = 1
d = 300
v = 1000
X = 1.0/(np.sqrt(d))*np.random.random([v,d])
w = 1.0/np.sqrt(d)*np.ones([d,1])
y_bar = np.matmul(X,w)
min_val = -1.0/np.sqrt(d)
max_val = 1.0/np.sqrt(d)

def randround(X,b):
    if b == 32:
        return torch.Tensor(X)
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
    l_i = np.linalg.eigvals(K)
    l_i = l_i * np.hstack([np.ones(d),np.zeros(v-d)])
    trace_val = sum([l**2/(l+lamb)**2 for l in l_i])
    R2 = 1.0/v*sigma*trace_val
    R = 0
    U,_,_ = np.linalg.svd(X_q)
    for i in range(v):
        R += (np.matmul(U[:,i],y_bar)/(l_i[i]+lamb))**2


    return 1.0/v*lamb**2*R + R2

b_s = [1,2,4,8,32]
lamb_s = [0.0000001 , 0.000001, 0.00001, 0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000]
results = np.zeros([len(b_s),len(lamb_s)])

for i in range(len(b_s)):
    b = b_s[i]
    X_q = randround(X,b)
    for j in range(len(lamb_s)):
        l = lamb_s[j]
        results[i,j] = compute_gen(X_q.data.numpy(), y_bar, sigma, l)


import matplotlib.pyplot as plt
for i in range(len(results)):
    plt.plot(lamb_s, results[i,:],label=str(b_s[i])+" bits")


plt.ylabel('loss')
plt.xlabel('lambda')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper left')
plt.show()
