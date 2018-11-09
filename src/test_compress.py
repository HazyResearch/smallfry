import unittest
import numpy as np
import utils
import compress

class CompressTest(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     cls.L = sparse.csr_matrix(np.array([[1, 0, 1]])

    def test_golden_section_search(self, verbose=False):
        '''Consider f(x) = (x-m)**2 + m for m in -10:10.  It's true minimum is
        at x = m.  Ensure that golden section search gets within specified
        tolerance of the true minimum.  Do this for various tolerances, and
        specify ranges to golden_section_search which either
        (1) include the true minimum in the interior,
        (2) include the true minimum on one of the edges, or
        (3) do not include the true minimum.
        For (1), ensure predicted minimum is within tolerance of true minimum.
        For (2), ensure predicted minimum is equal to true minimum.
        For (3), ensure predicted minimum is equal to edge closest to true minimum.
        '''
        tols=[1e-2,1e-3,1e-4]
        for tol in tols:
            for true_min in range(-10,11):
                f = lambda x : (x-true_min)**2 + true_min
                # try ranges where minimum is in the middle, on the edge, or
                # outside the specified range.
                ranges = [(-50,50), (true_min, 50), (-50,true_min), (true_min + 1,50), (-50, true_min - 1)]
                for r in ranges:
                    predicted_min = compress.golden_section_search(f,r[0],r[1],tol=tol)
                    if true_min > r[0] and true_min < r[1]:
                        # true_min is in the interior of the specified range.
                        self.assertTrue(np.abs(true_min - predicted_min) <= tol)
                    elif true_min == r[0] or true_min == r[1]:
                        # true_min is on the edge of the specified range.
                        self.assertEqual(true_min, predicted_min)
                    elif true_min + 1 == r[0]:
                        # true_min is below the specified range.
                        self.assertEqual(r[0], predicted_min)
                    elif true_min - 1 == r[1]:
                        # true_min is above the specified range.
                        self.assertEqual(r[1], predicted_min)
                    else:
                        # should never reach this code
                        self.assertTrue(False)
                    if verbose:
                        print('true min: {}, predicted min: {}'.format(true_min,predicted_min))

    def test_get_max_abs(self):
        ''' Test that get_max_abs correctly extracts the maximum absolute value
        from both 1D and 2D matrices, regardless of whether the value whose
        absolute value is largest is positive or negative.'''
        X1 = np.array([[-3,2],[2,4]])
        X2 = np.array([[-4,2],[2,3]])
        X3 = np.array([-4,2])
        X4 = np.array([4,2])
        self.assertEqual(compress.get_max_abs(X1), 4)
        self.assertEqual(compress.get_max_abs(X2), 4)
        self.assertEqual(compress.get_max_abs(X3), 4)
        self.assertEqual(compress.get_max_abs(X4), 4)


    def test_compress_uniform_internal(self, verbose=False):
        '''Test the internal function _compress_uniform.
        
        (1) Test for bit-rates 1,2, and for several range_limits, that a 
        hand-crafted matrix rounds each entry to the nearest centroid.
        (2) (a) Test that random data is always mapped to closest centroid
        when deterministic rounding is used (and (b) that this distance is at most
        half the distance b/w neighboring centroids). (c) When stochastic
        rounding is used, ensure that it is mapped to the closest or second
        closest centroid (and (d) that this distance is at most the distance b/w
        neighboring centroids). Do this for 1D and 2D data, for various 
        bit-rates, and for range_limits above and below the actual maximum
        absolute value of the matrix.
        (3) Test that if skip_quantize is True, that no quantization is done,
        regardless of stochastic_round or bit_rate.
        (4) Test that bitrate 0 always give a matrix of zeros
        when skip_quantize is False (regardless of stochastic round).
        (5) Test that when bitrate is 32, only the clipping is performed
        (no quantization), regardless of skip_quantize or stochastic_round.
        (6) Test that when range_limit is 0, compression always returns a 
        matrix of zeros, regardless of bit_rate, stochastic_round, or 
        skip_quantize.
        '''
        # (1) Test that for deterministic rounding with small bitrates, the
        # compressed values are equal to the hand calculated compression values.
        # We use bitrates 1 and 2, with range_limit set to the actual maximum
        # absolute value.  We include two options to allow for
        # "quantization ties" to go either way.  In particular, 0 is always
        # exactly in the middle of the two centroids on either side of it.
        for L in  [0.5,1,2]:
            # Data evenly spaced between -L and +L
            X = np.linspace(-L,L,21)
            X_b1_det_true = compress._compress_uniform(X,1,L,stochastic_round=False,skip_quantize=False)
            X_b2_det_true = compress._compress_uniform(X,2,L,stochastic_round=False,skip_quantize=False)
            X_b1_det_expected1 = np.array([-L]*11 + [L]*10)
            X_b1_det_expected2 = np.array([-L]*10 + [L]*11)
            X_b2_det_expected1 = np.array([-L]*4 + [-L/3]*7 + [L/3]*6 + [L]*4)
            X_b2_det_expected2 = np.array([-L]*4 + [-L/3]*6 + [L/3]*7 + [L]*4)
            self.assertTrue(np.allclose(X_b1_det_expected1, X_b1_det_true) or
                            np.allclose(X_b1_det_expected2, X_b1_det_true))
            self.assertTrue(np.allclose(X_b2_det_expected1, X_b2_det_true) or
                            np.allclose(X_b2_det_expected2, X_b2_det_true))

        # (2) (a) Test that random data is always mapped to closest centroid
        # when deterministic rounding is used (and (b) that this distance is at
        # most half the distance b/w neighboring centroids). (c) When stochastic
        # rounding is used, ensure that it is mapped to the closest or second
        # closest centroid (and (d) that this distance is at most the distance
        # b/w neighboring centroids). Do this for 1D and 2D data, for various 
        # bit-rates, and for range_limits above and below the actual maximum
        # absolute value of the matrix.
        X_dims = [(1000,1),(1000,100,1)]
        for L in  [0.5,1,2]:
            for r in [0.5*L, L, 2*L]:
                for b in [1,2,4,8]:
                    centroid_dims = [(1,2**b),(1,1,2**b)]
                    for i in range(len(X_dims)):
                        X_dim = X_dims[i]
                        centroid_dim = centroid_dims[i]
                        # Try 1D and 2D random data between -L and +L.
                        X = 2 * L * np.random.rand(*X_dim) - L
                        centroids = np.linspace(-r,r,2**b).reshape(centroid_dim)
                        # compute all pairwise distances between entries of X and 
                        # the list of centroids (This subtraction uses broadcasting)
                        pairwise_dist = np.abs(X - centroids)
                        # find distance between each entry of X and nearest centroid.
                        sort_dist = np.sort(pairwise_dist)
                        closest = sort_dist[...,0].reshape(X_dim)
                        Xq = compress._compress_uniform(X, b, r,
                                stochastic_round=False,
                                skip_quantize=False)
                        real_dist = np.abs(X-Xq)
                        # (2a) Assert that distance between each entry X[i,j] and nearest centroid
                        # is equal to the distance between X[i,j] and its quantized value Xq[i,j].
                        self.assertTrue(np.allclose(real_dist, closest))
                        # c is equal to the space between neighboring centroids.
                        c = 2*r/(2**b-1)
                        if r >= L:
                            # (2b) Assert, in the case where data is not clipped
                            # (r >= L), that each entry Xq[i,j] is at most
                            # c/2 away from X[i,j] (c is space b/w centroids)
                            self.assertTrue(np.all(real_dist <= c/2))

                        # (2c) For stochastic rounding, assert that the distance between
                        # X[i,j] and Xq[i,j] is equal to the distance between
                        # X[i,j] and the closest or second closest centroid.
                        second_closest = sort_dist[...,1].reshape(X_dim)
                        Xq = compress._compress_uniform(X, b, r,
                                stochastic_round=True,
                                skip_quantize=False)
                        real_dist = np.abs(X-Xq)
                        self.assertTrue(np.all(
                            np.isclose(real_dist,closest) + 
                            np.isclose(real_dist,second_closest) == True
                        ))
                        if r >= L:
                            # (2d) Assert, in the case where data is not clipped
                            # (r >= L), that each entry Xq[i,j] is at most
                            # c = 2*r/(2**b-1) away from X[i,j] (c is space b/w
                            # centroids). Also assert that the second closest
                            # centroid is at most c, and at least c/2, from
                            # every point, and that the closest centroid is at
                            # most c/2 from every point.
                            self.assertTrue(np.all(real_dist <= c))
                            self.assertTrue(np.all(second_closest <= c))
                            self.assertTrue(np.all(second_closest >= c/2))
                            self.assertTrue(np.all(closest <= c/2))

        # (3) Test that if skip_quantize is true, the compression function only
        # performs clipping, and does not perform quantization, regardless of
        # stochastic_round or bit_rate.
        for L in  [0.5,1,2]:
            for r in [0.5*L, L, 2*L]:
                for b in [0,1,2,4,8,32]:
                    for X_dim in X_dims:
                        for stochastic_round in [True,False]:
                            # Try 1D and 2D random data between -L and +L.
                            X = 2 * L * np.random.rand(*X_dim) - L
                            Xq = compress._compress_uniform(X, b, r,
                                stochastic_round=stochastic_round,
                                skip_quantize=True)
                            self.assertTrue(np.allclose(np.clip(X,-r,r), Xq))

        # (4) Test that bitrate 0 always gives a matrix of zeros
        # when skip_quantize is False (regardless of stochastic round).
        for L in  [0.5,1,2]:
            for r in [0.5*L, L, 2*L]:
                for X_dim in X_dims:
                    for stochastic_round in [True,False]:
                        # Try 1D and 2D random data between -L and +L.
                        X = 2 * L * np.random.rand(*X_dim) - L
                        Xq = compress._compress_uniform(X, 0, r,
                                stochastic_round=stochastic_round,
                                skip_quantize=False)
                        self.assertTrue(np.allclose(0,Xq))

        # (5) Test that when bitrate is 32, only the clipping is performed
        # (no quantizing), regardless of skip_quantize or stochastic_round.
        for L in  [0.5,1,2]:
            for r in [0.5*L, L, 2*L]:
                for X_dim in X_dims:
                    for stochastic_round in [True,False]:                        
                        for skip_quantize in [True,False]:
                            # Try 1D and 2D random data between -L and +L.
                            X = 2 * L * np.random.rand(*X_dim) - L
                            Xq = compress._compress_uniform(X, 32, r,
                                    stochastic_round=stochastic_round,
                                    skip_quantize=skip_quantize)
                            self.assertTrue(np.allclose(np.clip(X,-r,r), Xq))
                            

        # (6) Test that when range_limit is 0, compression always returns a 
        # matrix of zeros, regardless of bit_rate, stochastic_round, or 
        # skip_quantize.
        for L in  [0.5,1,2]:
            for r in [0.5*L, L, 2*L]:
                for b in [0,1,2,4,8,32]:
                    for X_dim in X_dims:
                        for stochastic_round in [True,False]:                        
                            for skip_quantize in [True,False]:
                                # Try 1D and 2D random data between -L and +L.
                                X = 2 * L * np.random.rand(*X_dim) - L
                                Xq = compress._compress_uniform(X, b, 0,
                                        stochastic_round=stochastic_round,
                                        skip_quantize=skip_quantize)
                                self.assertTrue(np.allclose(0, Xq))

        # (7) Test that quantization is unbiased for stochastic rounding.
        # We do this by constructing a random matrix 
        for L in  [0.5,1,2]:
            for b in [1,2,4,8]:
                for X_dim in X_dims:
                    # val is uniformly random in [-L,+L]
                    val = np.random.rand() * 2 * L - L
                    centroids = np.linspace(-L,L,2**b)
                    sort_dist = np.sort(np.abs(centroids-val))
                    # d1,d2 are the distances to closest/2nd closest centroids.
                    # the standard deviation of the quantization scheme is
                    # equal to np.sqrt(d1*d2).
                    d1,d2 = sort_dist[0],sort_dist[1]
                    stdev = np.sqrt(d1*d2)
                    # X is a matrix where every entry is equal to 'val'.
                    X = np.ones(X_dim) * val
                    Xq = compress._compress_uniform(X, b, L,
                            stochastic_round=True,
                            skip_quantize=False)
                    mean = np.mean(Xq)
                    n = np.prod(X.shape)
                    # By CLT z is roughly distributed as N(0,1)
                    c = 2*L/(2**b - 1)
                    z = (mean-val) * np.sqrt(n) / stdev
                    if verbose:
                        print('z = {}, b = {}, n = {}, normalized dist = {}'.format(
                            z, b, n, np.abs(mean-val)/(2*L)))
                    # with high probability, z is less than 4 standard
                    # deviations away from its mean.
                    self.assertTrue(np.abs(z) <= 4)

if __name__ == "__main__":
    unittest.main()