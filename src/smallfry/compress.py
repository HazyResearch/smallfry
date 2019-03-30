import os
import logging
import math
import time
import pathlib
import traceback
import numpy as np
from smallfry import utils

def compress_uniform(X, bit_rate, adaptive_range=False, stochastic_round=False,
        skip_quantize=False):
    '''
    This function compresses an embedding matrix using uniform quantization.

    Parameters:
        X (numpy array): Embedding matrix (rows of X are word embeddings).
        bit_rate (int): Number of bits to use per entry of the compressed embedding matrix.
        adaptive_range (bool): If True, golden section search is used to find the optimal
            value at which to clip the extreme values of the embedding matrix X before
            performing quantization.
        stochastic_round (bool): If True, stochastic rounding is used for the quantization.
        skip_quantize (bool): If True, the embedding matrix will not be quantized.
            If adaptive_range is True, the extreme values of X will still be clipped.

    Returns:
        Xq (numpy array): The compressed embedding matrix.
        frob_squared_error (float): The Frobenius norm of the difference between
            the compressed and uncompressed embedding matrices.
        elapsed (float): The duration (in seconds) of this function call.
    '''

    start = time.time()
    if adaptive_range:
        # Note that deterministic quantization is always uses for find_optimal_range.
        range_limit = find_optimal_range(X, bit_rate, stochastic_round=False)
    else:
        range_limit = get_max_abs(X)

    Xq = _compress_uniform(X, bit_rate, range_limit,
        stochastic_round=stochastic_round, skip_quantize=skip_quantize)
    elapsed = time.time() - start
    frob_squared_error = np.linalg.norm(X-Xq)**2
    return Xq, frob_squared_error, elapsed

# Internal function.  This one expects an explicit range_limit.
def _compress_uniform(X, bit_rate, range_limit, stochastic_round=False,
        skip_quantize=False):
    '''
    Internal uniform quantization function.

    Parameters:
        X (numpy array): Embedding matrix (rows of X are word embeddings).
        bit_rate (int): Number of bits to use per entry of the compressed embedding matrix.
        range_limit (float): All values in X with absolute value greater than
            this range_limit will be clipped.
        stochastic_round (bool): If True, stochastic rounding is used for the quantization.
        skip_quantize (bool): If True, the embedding matrix will not be quantized.
            If adaptive_range is True, the extreme values of X will still be clipped.

    Returns:
        Xq (numpy array): The compressed embedding matrix.
    '''
    assert range_limit >= 0, 'range_limit must be non-negative.'
    assert X.dtype == np.float or X.dtype == np.float64 or X.dtype == np.float32,\
                'Only floating point inputs allowed.'
    Xq = np.copy(X)
    if get_max_abs(Xq) > range_limit:
        np.clip(Xq, -range_limit, range_limit, out=Xq)
    if not skip_quantize and range_limit != 0:
        # We only need to quantize if skip_quantize is not set to true,
        # and range_limit != 0 (range_limit == 0 means the whole matrix is 
        # already set to 0)
        if bit_rate == 0:
            Xq[:] = 0
        elif bit_rate < 32:
            # affine transform to put Xq in [0,2**bit_rate - 1]
            Xq = (2**bit_rate - 1) * (Xq + range_limit) / (2 * range_limit)
            if stochastic_round:
                # each entry will round down if noise > fraction part
                np.ceil(Xq - np.random.rand(*Xq.shape), out=Xq)
            else:
                np.round(Xq, out=Xq)
            # undo affine transformation
            Xq = (Xq * 2 * range_limit) / (2**bit_rate - 1) - range_limit
        elif bit_rate >= 32:
            pass # don't quantize if bitrate >= 32
    return Xq

def find_optimal_range(X, bit_rate, stochastic_round=False, tol=1e-2):
    '''
    Find the best value to use to clip the embeddings before using uniform quantization.

    Parameters:
        X (numpy array): Embedding matrix (rows of X are word embeddings).
        bit_rate (int): Number of bits to use per entry of the compressed embedding matrix.
        stochastic_round (bool): If True, stochastic rounding is used for the quantization.
        tol (float): The tolerance (maximum possible error) for the golden section search
            algorithm.

    Returns:
        float: The optimal clipping value.
    '''
    f = lambda range_limit : compress_and_compute_frob_squared_error(
        X, bit_rate, range_limit, stochastic_round=stochastic_round)

    return golden_section_search(f, 0, get_max_abs(X), tol=tol)

def compress_and_compute_frob_squared_error(X, bit_rate, range_limit, stochastic_round=False):
    '''
    Function which computes frob squared error after compression.  This function
    is used in the find_optimal_range function to find best clip value for
    adaptive range uniform compression.

    Parameters:
        X (numpy array): Embedding matrix (rows of X are word embeddings).
        bit_rate (int): Number of bits to use per entry of the compressed embedding matrix.
        range_limit (float): All values in X with absolute value greater than
            this range_limit will be clipped.
        stochastic_round (bool): If True, stochastic rounding is used for the quantization.

    Returns:
        float: The squared Frobenius error which results from compressing the X matrix
            by first clipping its values to [-range_limit,range_limit], and then
            uniformly quantizing the clipped values within this range.
    '''
    Xq = _compress_uniform(X, bit_rate, range_limit, stochastic_round=stochastic_round)
    return np.linalg.norm(X - Xq)**2

def golden_section_search(f, x_min, x_max, tol=1e-2):
    '''
    Find argmin of f between x_min and x_max (for f uni-modal), to within a
    specified tolerance (tol), using the golden section search algorithm.

    Parameters:
        f (function): f is the unimodal function we would like to find the argmin
            for. f is assumed to take a scalar as input, and output a scalar.
        x_min (float): The minimum input to consider when minimizing f.
        x_max (float): The maximum input to consider when minimizing f.
        tol (float): The tolerance (maximum possible error) for the
            golden section search algorithm.

    Returns:
        float: The argmin of f, to within the specified tolerance.
    
    This function uses the golden-section search algorithm.
    It always maintains a list of four points [x1,x2,x3,x4],
    which are always spaced as: [a,a+(c^2)h,a+ch,a+h].
    for c = (math.sqrt(5) - 1) / 2 = 0.618...
    (c is equal to 1/phi, where phi = (1+sqrt(5))/2 is the golden ratio).
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
    # Initialize points
    # c is equal to 1/phi, for phi = (1+sqrt(5))/2
    c = (math.sqrt(5) - 1) / 2
    x1 = x_min
    x4 = x_max
    f_x1 = f(x1)
    f_x4 = f(x4)
    x2 = x1 + (x4-x1) * c**2
    x3 = x1 + (x4-x1) * c
    f_x2 = f(x2)
    f_x3 = f(x3)
    while (x4-x1 > tol):
        assert (math.isclose(x2, x1 + (x4 - x1) * c**2) and
                math.isclose(x3, x1 + (x4 - x1) * c))
        if f_x2 < f_x3:
            # The new points become [x1, NEW, x2, x3]
            x4,f_x4 = x3,f_x3
            x3,f_x3 = x2,f_x2
            x2 = x1 + (x4-x1) * c**2
            f_x2 = f(x2)
        else:
            # The new points become [x2, x3, NEW, x4]
            x1,f_x1 = x2,f_x2
            x2,f_x2 = x3,f_x3
            x3 = x1 + (x4-x1) * c
            f_x3 = f(x3)
        
    # Return x-value with minimum f(x) which was found.
    i = np.argmin([f_x1,f_x2,f_x3,f_x4])
    x = [x1,x2,x3,x4]
    return x[i]

def get_max_abs(X):
    return np.max(np.abs(X))
