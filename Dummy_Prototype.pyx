from cython cimport boundscheck, wraparound
cimport cython

# This method calculates the stresses on the boundary particles which influence the velocity and displacement of the
#  particles. Similarly to Adami et al. (2012) here, an averaged value of the neighborhood of fluid particles is
#  assigned to each dummy particle.
@boundscheck(False)
@wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef dummy_particles(double[:, ::1] mass, double[:, ::1] rho, double[:, ::1] r_ij, double[:, :, ::1] piola,
                      double[:, ::1] w, double[:, ::1] body, int[::1] part_type, long[:, ::1] ipn_pairs, long ipn):

    cdef:
        int row, col, pos, sign
        long idx, part, part_i, part_j, temp_part
        double[::1] wsum = mass[:, 0].copy()

    # Call kernel summation.
    bound_kernel_sum(mass, rho, w, part_type, ipn, ipn_pairs, wsum)

    for idx in range(1, ipn + 1):
        part_i = ipn_pairs[idx, 0]
        part_j = ipn_pairs[idx, 1]

        # Check for pairs with a boundary particle and a fluid particle.
        if part_type[part_i] != 1 and part_type[part_j] == 1:

            if wsum[part_i] > 0:
                for row in range(3):
                    for col in range(3):
                        piola[part_i, row, col] += ((piola[part_j, row, col] - rho[part_j, 0] * body[part_j, row] *
                                                     r_ij[idx, col]) * mass[part_j, 0] * w[idx, 0] / rho[part_j, 0]) / \
                                                   wsum[part_i]
                        # piola[part_i, row, col] += (piola[part_j, row, col] * mass[part_j, 0] * w[idx, 0] /
                        #                             rho[part_j, 0]) / wsum[part_i]

        elif part_type[part_i] == 1 and part_type[part_j] != 1:

            if wsum[part_j] > 0:
                for row in range(3):
                    for col in range(3):
                        piola[part_j, row, col] += ((piola[part_i, row, col] + rho[part_i, 0] * body[part_i, row] *
                                                     r_ij[idx, col]) * mass[part_i, 0] * w[idx, 1] / rho[part_i, 0]) / \
                                                   wsum[part_j]
                        # piola[part_j, row, col] += (piola[part_i, row, col] * mass[part_i, 0] * w[idx, 1] /
                        #                                 rho[part_i, 0]) / wsum[part_j]

# ======================================================================================================================
# Method that calculates the sum of the kernel for all dummy particles.
@boundscheck(False)
@wraparound(False)
@cython.cdivision(True)
cdef bound_kernel_sum(double[:, ::1] mass, double[:, ::1] rho, double[:, ::1] w, int[::1] part_type, long ipn,
                      long[:, ::1] ipn_pairs, double[::1] w_sum):

    cdef:
        long idx, part_i, part_j

    # Initialize the sums.
    w_sum[...] = 0

    for idx in range(1, ipn + 1):
        part_i = ipn_pairs[idx, 0]
        part_j = ipn_pairs[idx, 1]

        if part_type[part_i] != 1 and part_type[part_j] == 1:
            w_sum[part_i] += mass[part_j, 0] * w[idx, 0] / rho[part_j, 0]
        elif part_type[part_i] == 1 and part_type[part_j] != 1:
            w_sum[part_j] += mass[part_i, 0] * w[idx, 1] / rho[part_i, 0]

# ======================================================================================================================
