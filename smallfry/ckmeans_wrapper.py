# Wrapper for Ckmeans.1d.dp optimal univariate clustering library
# (C++ source from https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html)

import ctypes
import numpy as np
import os
import pathlib

cptr = ctypes.POINTER

def to_c_ptr(x, c_type):
  return x.ctypes.data_as(cptr(c_type))

def to_cint_ptr(x):
  return to_c_ptr(x, ctypes.c_int)

def to_cflt_ptr(x):
  return to_c_ptr(x, ctypes.c_float)

def to_cdbl_ptr(x):
  return to_c_ptr(x, ctypes.c_double)

class Ckmeans:
  def __init__(self, n_clusters, algorithm='linear', criterion='L2'):
    self.k = n_clusters
    self.algorithm = algorithm
    self.criterion = criterion
    if self.algorithm not in ('linear', 'loglinear'):
      raise ValueError('K-means DP algorithm must be one of: linear, loglinear')
    if self.criterion not in ('L1', 'L2'):
      raise ValueError('K-means DP criterion must be one of: L1, L2')
    path_2_thisdir = os.path.dirname(os.path.realpath(__file__))
    path_2_ckmeans = str(pathlib.PurePath(path_2_thisdir, 'ckmeans'))
    build_status = os.system('make -C %s 1> /dev/null' % path_2_ckmeans)
    if build_status != 0:
      raise OSError('Failed to build CKmeans library.')
    path_2_clib = str(pathlib.PurePath(path_2_ckmeans, 'bin/libckmeans.so'))
    ckmeans_lib = ctypes.cdll.LoadLibrary(path_2_clib)
    if ckmeans_lib.status() != 200:
      raise OSError('Failed to properly load CKmeans library.')
    self._kmeans_fn = ckmeans_lib.kmeans_dp
    self._kmeans_fn.restype = None
    self._kmeans_fn.argtypes = [cptr(ctypes.c_double), ctypes.c_int, ctypes.c_int,
        cptr(ctypes.c_int), cptr(ctypes.c_double), cptr(ctypes.c_double),
        cptr(ctypes.c_double), ctypes.c_bool, ctypes.c_bool]

  def fit(self, data):
    if data.dtype != ctypes.c_double:
      raise ValueError('Wrong input datatype.')
    k = self.k
    if len(data.shape) > 1:
      data = data.flatten()
    n = len(data)
    self.labels_ = np.empty(n, dtype=np.int32)
    self.cluster_centers_ = np.empty(k, dtype=np.float64)
    self.within_cluster_inertia_ = np.zeros(k, dtype=np.float64)
    self.cluster_sizes_ = np.empty(k, dtype=np.float64)
    self._kmeans_fn(to_cdbl_ptr(data), n, k, to_cint_ptr(self.labels_),
        to_cdbl_ptr(self.cluster_centers_), to_cdbl_ptr(self.within_cluster_inertia_),
        to_cdbl_ptr(self.cluster_sizes_), self.algorithm == 'loglinear',
        self.criterion == 'L2')
    self.inertia_ = np.sum(self.within_cluster_inertia_)
    self.cluster_centers_ = [[center] for center in self.cluster_centers_]
    return self