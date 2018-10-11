import torch
import numpy as np

def stochround(X,b,seed):
    '''
    Implements random uniform rounding over entire range [-L,L]
    L = max(abs(X))
    '''
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
    '''
    Implements deterministc midriser uniform quantization over entire range [-L,L]
    L = max(abs(X))
    '''
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

def uniquant(X,br,L):
        '''Copies X, quantizes X, returns X. Uses range L'''
        X_q = torch.Tensor(X)
        X_q = torch.clamp(X_q, min=-1*L, max=L)
        n = 2**br - 1
        X_q = (X_q+L)/(2*L)
        X_q = n*X_q # apply linear transform to put each quanta at integer
        X_q = torch.round(X_q)
        X_q = X_q/n #undo linear transform
        X_q = X_q*2*L - L #undo shift
        return X_q

def optranuni(X,br,eps=1e-40,tol=0.1,L_max=10):
    '''
    Implements the golden section line search
    Adaptively finds optimal range based on data
    Deterministic uniform rounding over optimal range
    '''
    br = int(br)
    quant = lambda X,L: uniquant(X,br,L) #bitrate does not change, no reason to pass it in each time
    f = lambda X,X_q: _compute_frobenius(X,X_q)
    L_star = _goldensearch(X,f,quant,eps=eps,tol=tol,L_max=L_max)
    return quant(X,L_star)

def clipnoquant(X,br):
    br = int(br)
    quant = lambda X,L: uniquant(X,br,L) #bitrate does not change, no reason to pass it in each time
    f = lambda X,X_q: _compute_frobenius(X,X_q)
    L_star = _goldensearch(X,f,quant)
    return torch.clamp(torch.Tensor(X), min=-1*L_star, max=L_star)

def _goldensearch(X,f,quant,eps=1e-40,tol=0.1,L_max=10):
    '''
    Implements the golden section line search
    Adaptively finds optimal range based on data
    '''
    phi = (1+np.sqrt(5))/2 # golden ratio
    #initialize line search iteration
    a = eps
    b = L_max
    val_a = f(X,quant(X,a))
    val_b = f(X,quant(X,b))
    c = b - (b-a)/phi
    d = a + (b-a)/phi
    val_c = f(X,quant(X,c))
    val_d = f(X,quant(X,d))
    #perform iterations
    while (b-a > tol):
        if val_c < val_d:
            b = d
            val_b = val_d
            d = c
            val_c = val_d
            c = b - (b-a)/phi
            
        else:
            a = c
            val_a = val_c
            c = d
            val_c = val_d
            d = a + (b-a)/phi
        val_c = f(X,quant(X,c))
        val_d = f(X,quant(X,d))
    #on termination, return optimal range
    return c if val_c < val_d else d

def _compute_frobenius(baseX,X_q):
        '''Value we are minimizing -- Frobenius distance'''
        return np.linalg.norm(baseX-X_q)
