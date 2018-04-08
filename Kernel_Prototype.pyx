from libc.math cimport pi, pow
cimport cython
from cython cimport boundscheck, wraparound

"""This module implements the cubic spline, Wendland C6, and quintic spline kernels, as presented in Dehnen and Aly 
    (2012): Improving convergence in smoothed particle hydrodynamics simulations without pairing instability. """

# ======================================================================================================================
# This function returns the normalization factor for each of the supported kernels.
@cython.cdivision(True)
@cython.profile(True)
cdef double normalization_factor(int kernel_type, double r, int sim_dim):

    if sim_dim == 3:

        # Cubic spline.
        if kernel_type == 1:
            return 16 / (pi * pow(r, 3))

        # Wendland C6.
        elif kernel_type == 2:
            return 1365 / (64 * pi * pow(r, 3))

        # Quintic spline.
        elif kernel_type == 3:
            return 216 / (960 * pi * pow(r, 3))

    elif sim_dim == 2:
        if kernel_type == 1:
            return 80 / (7 * pi * pow(r, 2))
        elif kernel_type == 2:
            return 78 / (7 * pi * pow(r, 2))
        elif kernel_type == 3:
            return 252 / (1912 * pi * pow(r, 2))

    else:
        if kernel_type == 1:
            return 8 / (3 * r)
        elif kernel_type == 3:
            return 6 / (240 * r)

# ======================================================================================================================
# This function calculates the kernel and kernel derivatives.
@boundscheck(False)
@wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef kernel_calcs(double[:, ::1] r_ij, double[:, ::1] rij, double h, int sim_dim, double[:, ::1] w,
                   double[:, :, ::1] dw, int kernel_type, long ipn):
    cdef:
        int i
        long idx
        double a, b, c, q, r = 2 * h    # Kernel radius.
        double norm_factor = normalization_factor(kernel_type, r, sim_dim)

    # Cubic spline kernel.
    if kernel_type == 1:
        for idx in range(1, ipn + 1):

            # Non-dimensional distance.
            q = rij[idx, 0] / r

            if q <= 0.5:
                w[idx, 0] = pow(1 - q, 3) - 4 * pow(0.5 - q, 3)

                if q > 0:
                    for i in range(3):
                        dw[idx, 0, i] = -(3 * pow(1 - q, 2) - 12 * pow(0.5 - q, 2)) * r_ij[idx, i] / (rij[idx, 0] * r)

            elif 0.5 < q < 1.0:
                w[idx, 0] = pow(1 - q, 3)

                for i in range(3):
                    dw[idx, 0, i] = -(3 * pow(1 - q, 2)) * r_ij[idx, i] / (rij[idx, 0] * r)

            # Apply normalization factor.
            w[idx, 0] *= norm_factor
            w[idx, 1] = w[idx, 0]
            dw[idx, 0, 0] *= norm_factor
            dw[idx, 0, 1] *= norm_factor
            dw[idx, 0, 2] *= norm_factor
            dw[idx, 1, 0] = -dw[idx, 0, 0]
            dw[idx, 1, 1] = -dw[idx, 0, 1]
            dw[idx, 1, 2] = -dw[idx, 0, 2]

    # This kernel formulation is incorrect, please revise!
    # Quintic spline kernel.
    elif kernel_type == 3:
        for idx in range(1, ipn + 1):

            # Non-dimensional distance.
            q = rij[idx, 0] / r

            if q <= 1.0 / 3.0:
                w[idx, 0] = pow(3 - 3 * q / 4.0, 5) - 6 * pow(2 - 3 * q / 4.0, 5) + 15 * pow(1 - 3 * q / 4.0, 5)

                if q > 0:
                    for i in range(3):
                        dw[idx, 0, i] = -(pow(3 - 3 * q / 4.0, 4) - 6 * pow(2 - 3 * q / 4.0, 4) + 15 *
                                          pow(1 - 3 * q / 4.0, 4)) * r_ij[idx, i] / (rij[idx, 0] * h)

            elif 1.0 / 3.0 < q <= 2.0 / 3.0:
                w[idx, 0] = pow(3 - 3 * q / 4.0, 5) - 6 * pow(2 - 3 * q / 4.0, 5)

                for i in range(3):
                    dw[idx, 0, i] = -(pow(3 - 3 * q / 4.0, 4) - 6 * pow(2 - 3 * q / 4.0, 4)) * r_ij[idx, i] / \
                                    (rij[idx, 0] * h)

            elif 2.0 / 3.0 < q < 1.0:
                w[idx, 0] = pow(3 - 3 * q / 4.0, 5)

                for i in range(3):
                    dw[idx, 0, i] = -pow(3 - 3 * q / 4.0, 4) * r_ij[idx, i] / (rij[idx, 0] * h)

            # Apply normalization factor.
            w[idx, 0] *= norm_factor
            w[idx, 1] = w[idx, 0]
            dw[idx, 0, 0] *= norm_factor
            dw[idx, 0, 1] *= norm_factor
            dw[idx, 0, 2] *= norm_factor
            dw[idx, 1, 0] = -dw[idx, 0, 0]
            dw[idx, 1, 1] = -dw[idx, 0, 1]
            dw[idx, 1, 2] = -dw[idx, 0, 2]

    # Wendland C6 kernel.
    else:
        for idx in range(1, ipn + 1):

            # Non-dimensional distance.
            q = rij[idx, 0] / r

            if q < 1.0:
                # Auxiliary variables.
                a = 1 - q
                b = 1 + 8 * q + 25 * pow(q, 2) + 32 * pow(q, 3)

                w[idx, 0] = pow(a, 8) * b

                if q > 0:
                    # Auxiliary variable, c = db/dq.
                    c = 8 + 50 * q + 96 * pow(q, 2)

                    for i in range(3):
                        dw[idx, 0, i] = -(8 * pow(a, 7) * b - pow(a, 8) * c) * r_ij[idx, i] / (rij[idx, 0] * r)

            # Apply normalization factor.
            w[idx, 0] *= norm_factor
            w[idx, 1] = w[idx, 0]
            dw[idx, 0, 0] *= norm_factor
            dw[idx, 0, 1] *= norm_factor
            dw[idx, 0, 2] *= norm_factor
            dw[idx, 1, 0] = -dw[idx, 0, 0]
            dw[idx, 1, 1] = -dw[idx, 0, 1]
            dw[idx, 1, 2] = -dw[idx, 0, 2]

# ======================================================================================================================
# This method gives the value of the kernel at the particle and at a distance equal to the initial interparticle
#  distance, both used in the artificial stress calculation.
@boundscheck(False)
@wraparound(False)
@cython.profile(True)
cpdef calc_ref_kernel(double[:, ::1] wref, double dp, double h, int sim_dim, int kernel_type):

    cdef:
        double a[2][3]
        double b[2][2]
        double c[2][2][3]
        double[:, ::1] r_ij_ref = a
        double[:, ::1] rij_ref = b
        double[:, :, ::1] dw_ref = c

    # Initialize values.
    r_ij_ref[...] = 0
    r_ij_ref[1, 0] = dp
    rij_ref[...] = 0
    rij_ref[1, 0] = dp
    dw_ref[...] = 0

    kernel_calcs(r_ij_ref, rij_ref, h, sim_dim, wref, dw_ref, kernel_type, 2)
# ======================================================================================================================
