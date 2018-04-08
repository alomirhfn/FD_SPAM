from cython cimport boundscheck, wraparound
cimport cython

# Method to calculate the impact forces based on the simple summation of traction vectors on the boundary. This is
#  method 1 of Favero Neto and Borja (2018).
@boundscheck(False)
@wraparound(False)
@cython.profile(True)
cpdef impact_force1(int[::1] part_type, double[:, :, ::1] stress, double[::1] normal, double dp, long num_parts,
                    double[::1] force):

    # Auxiliary variables.
    cdef int part, row, col

    # Initialize force vector.
    force[...] = 0

    for part in range(1, num_parts + 1):
        if part_type[part] == 2:

            for row in range(3):
                for col in range(3):
                    force[row] += stress[part, row, col] * normal[col] * dp

# ======================================================================================================================
# Method to calculate the impact forces based on the summation of traction vectors on the boundary. These vectors are
#  obtained from the smoothing of the stress on the boundary. This is method 2 of Favero Neto and Borja (2018).
@boundscheck(False)
@wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef impact_force2(double[:, :, ::1] stress, double[:, ::1] w, int[::1] part_type, long[:, ::1] ipn_pairs, long ipn,
                    long num_parts, double dp, double[::1] normal, double[::1] force, double[::1] wsum):

    cdef:
        int row, col
        long idx, part, part_i, part_j
        double[:, :, ::1] temp = stress.copy()

    temp[...] = 0
    wsum[...] = 0
    force[...] = 0

    # Call kernel summation.
    bound_kernel_sum(w, part_type, ipn, ipn_pairs, wsum)

    for idx in range(1, ipn + 1):

        part_i = ipn_pairs[idx, 0]
        part_j = ipn_pairs[idx, 1]

        # Check for boundary particle (part_i != 1) and fluid particle (part_j == 1).
        if part_type[part_i] == 2 and part_type[part_j] == 2 and part_i != part_j:

            if wsum[part_i] != 0 and wsum[part_j] != 0:
                for row in range(3):
                    for col in range(3):
                        temp[part_i, row, col] += stress[part_j, row, col] * w[idx, 0] / wsum[part_i]
                        temp[part_j, row, col] += stress[part_i, row, col] * w[idx, 1] / wsum[part_j]

            elif wsum[part_i] != 0:
                for row in range(3):
                    for col in range(3):
                        temp[part_i, row, col] += stress[part_j, row, col] * w[idx, 0] / wsum[part_i]

            elif wsum[part_j] != 0:
                for row in range(3):
                    for col in range(3):
                        temp[part_j, row, col] += stress[part_i, row, col] * w[idx, 1] / wsum[part_j]

        elif part_type[part_i] == 2 and part_type[part_j] == 2 and part_i == part_j:
            if wsum[part_i] != 0:
                for row in range(3):
                    for col in range(3):
                        temp[part_i, row, col] += stress[part_j, row, col] * w[idx, 0] / wsum[part_i]

    stress[...] = temp[...]

    # Calculate the resulting force.
    impact_force1(part_type, stress, normal, dp, num_parts, force)

# ======================================================================================================================
# Method that calculates the sum of the kernel for all dummy particles.
@boundscheck(False)
@wraparound(False)
@cython.cdivision(True)
cdef bound_kernel_sum(double[:, ::1] w, int[::1] part_type, long ipn, long[:, ::1] ipn_pairs, double[::1] w_sum):

    cdef:
        long idx, part_i, part_j

    # Initialize the sums.
    w_sum[...] = 0

    for idx in range(1, ipn + 1):

        part_i = ipn_pairs[idx, 0]
        part_j = ipn_pairs[idx, 1]

        if part_type[part_i] == 2 and part_type[part_j] == 2 and part_i != part_j:
            w_sum[part_i] += w[idx, 0]
            w_sum[part_j] += w[idx, 1]

        elif part_i == part_j:
            w_sum[part_i] += w[idx, 0]

# ======================================================================================================================