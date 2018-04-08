from __future__ import print_function
__author__ = 'alomir'

from Timer import Timer
import numpy as np
from cython cimport boundscheck, wraparound
cimport cython

t = Timer()

# ======================================================================================================================
# This method returns the sum of the kernels in the domain, used to calculate smoothing of variables and for the
#  so-called Shepard filter, which enforces zero-th order consistency for the kernel function.
@boundscheck(False)
@wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef kernel_sum(int[::1] part_type, double[:, ::1] mass, double[:, ::1] rho, int[:, ::1] ipn_pairs, long ipn,
                 double[:, ::1] w, double[::1] w_sum):

    t.tic()
    cdef long part_i, part_j, idx

    # Initialize memoryview.
    w_sum[...] = 0

    for idx in range(1, ipn + 1):
        part_i = ipn_pairs[idx, 0]
        part_j = ipn_pairs[idx, 1]

        # Consider only interactions between fluid parts or boundary and fluid particles (no boundary-boundary).
        if (part_type[part_i] == 1 or part_type[part_j] == 1) and part_i != part_j:
            w_sum[part_i] += mass[part_j, 0] * w[idx, 0] / rho[part_j, 0]
            w_sum[part_j] += mass[part_i, 0] * w[idx, 1] / rho[part_i, 0]
        elif part_type[part_i] == 1 and part_i == part_j:
            w_sum[part_i] += mass[part_j, 0] * w[idx, 0] / rho[part_j, 0]

    t.toc('Kernel sum Cython')

# ======================================================================================================================
# This method returns the sum of the kernel derivatives for every fluid and dummy particle with fluid neighbors and the
#  matrix M, whose inverse is used to restore first-order consistency.
@boundscheck(False)
@wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef double[:, ::1] kernel_deriv_moments(int[::1] part_type, double[:, ::1] mass, double[:, ::1] rho,
                                          double[:, ::1] r_ij, long ipn, int[:, ::1] ipn_pairs, double[:, :, ::1] dw,
                                          double[:, :, ::1] m):

    """ Note: matrix L is the inverse of matrix M and that is the matrix used to correct the kernel derivative"""
    t.tic()
    cdef:
        int row, col
        long idx, part_i, part_j
        double[:, ::1] dw_sum = r_ij.copy()

    # Initializing the sum of the kernel derivatives and matrix M.
    dw_sum[...] = 0
    m[...] = 0

    for idx in range(1, ipn + 1):

        part_i = ipn_pairs[idx, 0]
        part_j = ipn_pairs[idx, 1]

        if part_type[part_i] == 1 or part_type[part_j] == 1:

            for row in range(3):
                dw_sum[part_i, row] += mass[part_j, 0] * dw[idx, 0, row] / rho[part_j, 0]
                dw_sum[part_j, row] += mass[part_i, 0] * dw[idx, 1, row] / rho[part_i, 0]
                for col in range(3):
                    m[part_i, row, col] -= mass[part_j, 0] * r_ij[idx, row] * dw[idx, 0, col] / rho[part_j, 0]
                    m[part_j, row, col] += mass[part_i, 0] * r_ij[idx, row] * dw[idx, 1, col] / rho[part_i, 0]

    t.toc('Kernel derivative momentums Cython')
    return dw_sum

# ======================================================================================================================
# This method renormalizes the kernel so that it restores zero-th and first order consistency. The formulation was
#  taken from: Variational and momentum preservation aspects of Smooth Particle Hydrodynamic formulations - Bonet
#  and Lok (1999).
@boundscheck(False)
@wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef shepard_filter(long ipn, int[:, ::1] ipn_pairs , double[:, ::1] w, double[::1] w_sum):

    t.tic()
    cdef long idx, part_i, part_j

    for idx in range(1, ipn + 1):

        part_i = ipn_pairs[idx, 0]
        part_j = ipn_pairs[idx, 1]

        if w_sum[part_i] > 0:
            w[idx, 0] /= w_sum[part_i]

        if w_sum[part_j] > 0:
            w[idx, 1] /= w_sum[part_j]

    t.toc('Shepard filter Cython')

# ======================================================================================================================
# This method restores 1st order consistency of the kernel derivative. This formulation was presented in Bonet and Lok
# (1999) and Violeau (2012).
@boundscheck(False)
@wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef kernel_grad_correction(long ipn, int[:, ::1] ipn_pairs, double[:, :, ::1] dw, double[:, :, ::1] m, int sim_dim,
                             long num_parts, double tol):

    t.tic()
    cdef:
        int row, col
        long idx, part_i, part_j
        double[:, :, ::1] dw_temp = dw.copy()
        double[:, :, ::1] g = m.copy()
        double[:, ::1] g_temp

    # Initialize the matrix L.
    g[...] = 0
    dw_temp[...] = 0

    # Transforms to Numpy array and eliminate small residues.
    np_m = np.asarray(m)
    np_m[np.abs(np_m) <= tol] = 0

    # This is so the matrix has no zeros in diagonal in case of 1D and 2D simulations.
    if sim_dim == 1:
        np_m[:, 1, 1] = np_m[:, 2, 2] = 1
    elif sim_dim == 2:
            np_m[:, 2, 2] = 1

    for part in range(1, num_parts + 1):
        if np.linalg.det(np_m[part]) > 1e-6:
            g_temp = np.linalg.inv(np_m[part])
            g[part, ...] = g_temp[...]
        else:
            g[part, 0, 0] = 1
            g[part, 1, 1] = 1
            g[part, 2, 2] = 1

    for idx in range(1, ipn + 1):
        part_i = ipn_pairs[idx, 0]
        part_j = ipn_pairs[idx, 1]

        for row in range(3):
            for col in range(3):
                dw_temp[idx, 0, row] += dw[idx, 0, col] * g[part_i, col, row]
                dw_temp[idx, 1, row] += dw[idx, 1, col] * g[part_j, col, row]

    dw[...] = dw_temp[...]
    t.toc('Kernel gradient correction Cython')

# ======================================================================================================================
