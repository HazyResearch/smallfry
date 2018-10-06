import torch

def stochround(X,b,seed):
    b = int(b)
    torch.manual_seed(seed)
    dtype = torch.cuda.FloatTensor
    X = torch.Tensor(X)
    L = torch.max( torch.abs( X )) # compute range
    n = 2**b - 1
    X = X / (2*L) # apply affine transform to get on unit interval
    X = X+0.5
    X = n*X # apply linear transform to put each quanta at integer
    X = X - torch.rand(X.shape) # each entry will round down if noise > fraction part
    X = torch.ceil(X)
    X = X/n #undo linear transform
    X = X-0.5 #undo shift
    X = X*2*L #put back in original range
    return X

def midriser(X,b):
    b = int(b)
    delta = 1/2**b
    dtype = torch.cuda.FloatTensor
    X = torch.Tensor(X)
    eps = 1e-5
    L = torch.max( torch.abs( X )) + eps # compute range
    X = X / (2*L) # apply affine transform to get on unit interval
    X = X+0.5
    X = delta*( torch.floor(X/delta) + 0.5)
    X = X - 0.5
    X = X*2*L
    return X
