import torch
import numpy as np
from scipy.interpolate import interp1d

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
    '''Copies X, quantizes X, returns X. Uses range L and bitrate br'''
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
    if br == 32:
        return X
    quant = lambda X,L: uniquant(X,br,L) #bitrate does not change, no reason to pass it in each time
    f = lambda X,X_q: _compute_frobenius(X,X_q)
    L_star = _goldensearch(X,f,quant,eps=eps,tol=tol,L_max=L_max)
    X_q = quant(X,L_star)
    return X_q.numpy()

def stochoptranuni(X,br,seed=1234,eps=1e-40,tol=0.1,L_max=10):
    '''
    Implements the golden section line search
    Adaptively finds optimal range based on data
    Deterministic uniform rounding over optimal range
    '''
    br = int(br)
    if br == 32:
        return X
    quant = lambda X,L: uniquant(X,br,L) #bitrate does not change, no reason to pass it in each time
    f = lambda X,X_q: _compute_frobenius(X,X_q)
    L_star = _goldensearch(X,f,quant,eps=eps,tol=tol,L_max=L_max)
    X_q = clamp_and_quantize(X, br, range_limit=L_star, stochastic_round=True)
    return X_q.numpy()

def clipnoquant(X,br):
    '''
    Clips where it would clip with oprtanuni, but otherwise full-precision 
    '''
    br = int(br)
    quant = lambda X,L: uniquant(X,br,L) #bitrate does not change, no reason to pass it in each time
    f = lambda X,X_q: _compute_frobenius(X,X_q)
    L_star = _goldensearch(X,f,quant)
    X_clip = torch.clamp(torch.Tensor(X), min=-1*L_star, max=L_star)
    return X_clip.numpy()

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

'''
This method clamps X between -range_limit and +range_limit, and then quantizes
X into one of 2**br possible values, using mid_riser quantization values or not,
and using stochastic or deterministic rounding.
If bit_rate == 32, no quantization is done. 
If range_limit == np.inf, no clamping is done.
QUESTION: DO WE WANT TO CHANGE X IN-PLACE?
'''
def clamp_and_quantize(X, bit_rate=32, range_limit=np.inf, use_midriser=False, stochastic_round=False):
    assert range_limit >= 0, 'range_limit must be non-negative.'
    do_clamp = range_limit != np.inf
    do_quantize = bit_rate < 32
    if range_limit == np.inf:
        range_limit = get_max_abs(X)
    if do_quantize and use_midriser:
        # The only difference between 'midriser' and 'not midriser' is that 'midriser'
        # picks the 2**bit_rate quantization values between 
        # [-range_limit + r, range_limit - r] for r = range_limit/2**bit_rate,
        # while 'not midriser' chooses the 2**bit_rate quantization values between
        # [-range_limit,+range_limit].
        range_limit -= range_limit / 2**bit_rate
    X_q = torch.tensor(X) # creates a copy of X
    if use_midriser or do_clamp:
        # When using mid-riser, always need to clamp.
        # If not using mid-riser, only need to clamp if user specified a range_limit.
        X_q = torch.clamp(X_q, min=-range_limit, max=range_limit)
    if do_quantize:
        X_q = quantize(X_q, bit_rate, range_limit, stochastic_round=stochastic_round)
        #X_q = quantize_with_scipy(X_q,range_limit, bit_rate, stochastic_round=stochastic_round)
    return X_q

'''
X is a Tensor where all entries are between -range_limit and +range_limit.
Letting L=range_limit, and r = 2*L/(2**bit_rate-1), this method quantizes X 
into one of the values in {-L,-L+r,-L+2r,...,+L} (a set of size 2**bit_rate)
QUESTION: DO WE WANT TO CHANGE X IN-PLACE?
'''
def quantize(X, bit_rate, range_limit, stochastic_round=False):
    assert range_limit != np.inf and range_limit >= 0, 'range_limit must be finite and non-negative.'
    assert get_max_abs(X) <= range_limit, 'X must be between -range_limit and +range_limit'
    assert bit_rate < 32, 'Only bit_rates < 32 supported.'
    # affine transform to put X in [0,2**bit_rate - 1]
    X_q = (2**bit_rate - 1) * (X + range_limit) / (2 * range_limit) # not in-place
    if stochastic_round:
        X_q = X_q - torch.rand(X_q.shape)
        # each entry will round down if noise > fraction part
        X_q = torch.ceil(X_q)
    else:
        X_q = torch.round(X_q)
    # undo affine transformation
    X_q = (X_q * 2 * range_limit) / (2**bit_rate - 1) - range_limit 
    return X_q

''' 
Quantize X just like the above method, but using scipy.interpolate.interp1d.
QUESTION: DO WE WANT TO CHANGE X IN-PLACE?
***NOT TESTED***
'''
def quantize_with_scipy(X, bit_rate, range_limit, stochastic_round=False):
    assert range_limit != np.inf, 'range_limit must be finite.'
    assert get_max_abs(X) <= range_limit, 'X must be between -range_limit and +range_limit'
    assert bit_rate < 32, 'Only bit_rates < 32 supported.'
    bin_edges = np.linspace(-range_limit, range_limit, 2**bit_rate)
    bin_size = 2 * range_limit / (2**bit_rate - 1)
    X_q = torch.tensor(X) # creates a copy of X
    if stochastic_round:
        X_q -= torch.rand(X_q.shape) * bin_size
        interp = interp1d(bin_edges,bin_edges,kind='next') # fill_value='extrapolate'
    else:
        interp = interp1d(bin_edges,bin_edges,kind='nearest') # fill_value='extrapolate'
    X_q = torch.from_numpy(interp(X_q.numpy()))
    return X_q

def get_max_abs(X):
    return torch.max(torch.abs(X)).item()

# TESTS
def test1():
    X = torch.tensor([-1.5,-0.5,0.5,1.5])
    X_expect = torch.tensor([-1.0,-1.0,1.0,1.0])
    Xq = clamp_and_quantize(X,bit_rate=1,range_limit=1)
    assert torch.all(torch.eq(Xq, X_expect)).item() == 1

def test2():
    X = torch.tensor([-1.5,-0.5,0.5,1.5])
    X_expect = X
    Xq = clamp_and_quantize(X,bit_rate=2)
    assert torch.all(torch.eq(Xq, X_expect)).item() == 1

def test3():
    X = torch.tensor([-1.5,-0.5,0.5,1.5])
    X_expect = X
    Xq = clamp_and_quantize(X,bit_rate=2,range_limit=2,use_midriser=True)
    assert torch.all(torch.eq(Xq, X_expect)).item() == 1
