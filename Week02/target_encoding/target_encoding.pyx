# distutils: language = c++

import numpy as np
cimport numpy as np
import pandas as pd
cimport cython

cdef extern from "/content/drive/MyDrive/target_encoding/target_encoding.h":
  void target_mean(double *matrix, double *result, const long row, const long col, const long x_index, const long y_index)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef target_mean_v5(data, y_name, x_name):

  cdef long length = data.shape[0]
  cdef long x_index = data.columns.get_loc(x_name)
  cdef long y_index = data.columns.get_loc(y_name)

  cdef np.ndarray[np.float64_t, ndim=1, mode = 'c'] result = np.ascontiguousarray(np.zeros(length), dtype=np.float64)
  cdef np.ndarray[np.float64_t, ndim=2, mode = 'c'] matrix = np.ascontiguousarray(data[[y_name, x_name]].values, dtype=np.float64)

  cdef double* result_buff = <double*> result.data
  cdef double* matrix_buff = <double*> matrix.data

  cdef long row = matrix.shape[0]
  cdef long col = matrix.shape[1]
  target_mean(matrix_buff, result_buff, row, col, x_index, y_index)

  return result
