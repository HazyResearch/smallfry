import torch
import numpy as np
import math
from scipy.interpolate import interp1d

'''
HIGH-LEVEL QUANTIZATION CALLS
'''
def stochround(X,b):
    return uniform_quantizer(X, b, _fullrange, _stochround)
def naiveuni(X,b):
    return uniform_quantizer(X, b, _fullrange, _round)
def adarange(X,b):
    return uniform_quantizer(X, b, _adarange, _round)
def stoch_adarange(X,b):
    return uniform_quantizer(X, b, _adarange, _stochround)
def clip_no_quant(X,b):
    return uniform_quantizer(X, b, _adarange, lambda Y: Y)
def midriser(X,b):
    raise ValueError("Midriser is no longer supported")
def stoch_adarange_2(X,b):
    return uniform_quantizer(X, b, _adarange_rand, _stochround)
    
'''
CORE QUANTIZER
'''
def uniform_quantizer(X, b, q_range, quantize):
    b = int(b)
    X = torch.Tensor(X)
    L = q_range(X, b)
    forward_map = lambda Y: _affine_transform(Y,L,b)
    backward_map = lambda Y: _affine_transform(Y,L,b,invert=True)
    return backward_map(quantize(forward_map(_clip(X,L)))).numpy()

'''
Range solvers
'''
def _adarange(X,b,stochastic=False):    
    return golden_section_search(lambda L : np.linalg.norm(X.numpy() - uniform_quantizer(
        X, b, lambda Y,z : L, _adarange_rand if stochastic else _round)))

def _adarange_rand(X,b):
    return _adarange(X,b,stochastic=True)

def _fullrange(X,b):
    return torch.max(torch.abs(X))

'''
Rounding schemes
'''
def _stochround(X):
    return torch.ceil(X - torch.rand(X.shape))

def _round(X):
    return torch.round(X)

'''
HELPERS
'''
def _affine_transform(X,L,b,invert=False):
    n = 2**b-1
    interval = 2*L
    shift = 0.5
    return (X/n - shift)*interval if invert else  n*(X/interval + shift)

def _clip(X,L):
    eps = 1e-40
    L -= eps
    return torch.clamp(X, min=-1*L, max=L)

def golden_section_search(f, x_min=1e-5, x_max=10, tol=1e-2):
    '''
    Find argmin of f between x_min and x_max (for f uni-modal).
    
    This function uses the golden-section search algorithm.
    It always maintains a list of four points [x1,x2,x3,x4],
    which are always spaced as: [a,a+(c^2)h,a+ch,a+h].
    for c = (math.sqrt(5) - 1) / 2 = 0.618...
    The algorithm progressively reduces the size of the interval being
    considered by checking whether f(x2) < f(x3), and eliminating one of the
    endpoints accordingly; x4 is eliminated if f(x2) < f(x3), and x1 
    is eliminated otherwise.
    
    If f(a+(c^2)h) < f(a+ch), the new interval becomes
    >>> [a,a+(c^3)h,a+(c^2)h,a+ch] = [a,a+(c^2)(ch),a+c(ch),a+ch]
    (So h' = ch, a' = a)
    Otherwise, the interval becomes
    >>> [a',a'+(c^2)h',a'+ch', a'+h'], for a' = a+(c^2)h and h'=(h-(c^2)h)
    It is easy to check that a'+(c^2)h' = a + ch, and that a'+h' = a+h,
    So this interval is equal to [a+(c^2)h, a+ch, X, a+h], for X=a'+ch'

    The algorithm terminates when it has been narrowed
    down that the argmin must be in an interval of size < tol.
    '''
    #initialize points
    phi = (math.sqrt(5) - 1) / 2
    x1 = x_min
    x4 = x_max
    f_x1 = f(x1)
    f_x4 = f(x4)
    x2 = x1 + (x4-x1) * phi**2
    x3 = x1 + (x4-x1) * phi
    f_x2 = f(x2)
    f_x3 = f(x3)
    while (x4-x1 > tol):
        assert (math.isclose(x2, x1 + (x4 - x1) * phi**2) and 
                math.isclose(x3, x1 + (x4 - x1) * phi))
        if f_x2 < f_x3:
            # The new points become [x1, NEW, x2, x3]
            x4,f_x4 = x3,f_x3
            x3,f_x3 = x2,f_x2
            x2 = x1 + (x4-x1) * phi**2
            f_x2 = f(x2)
        else:
            # The new points become [x2, x3, NEW, x4]
            x1,f_x1 = x2,f_x2
            x2,f_x2 = x3,f_x3
            x3 = x1 + (x4-x1) * phi
            f_x3 = f(x3)
        
    # Return x-value with minimum f(x) which was found.
    i = np.argmin([f_x1,f_x2,f_x3,f_x4])
    x = [x1,x2,x3,x4]
    return x[i]

