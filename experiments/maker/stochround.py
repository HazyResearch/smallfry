import torch

def stochround(X,b):
    L = torch.max( torch.abs( X )) # compute range
    n = 2**b - 1
    quanta = [] # compute quanta
    for i in range(n+1):
        quanta.append(i/n)
    quanta = torch.Tensor(quanta)
    X = X / (2*L) # apply affine transform to get on unit interval
    X = X+0.5
    X = n*X # apply linear transform to put each quanta at integer
    X = X - torch.rand(X.shape) # each entry will round down if noise > fraction part
    X = torch.ceil(X)
    X = X/n #undo linear transform
    X = X-0.5 #undo shift
    X = X*2*L #put back in original range
    return X
