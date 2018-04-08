from cython cimport boundscheck, wraparound
cimport cython

# This method calculates the rates of change of momentum and energy.
@boundscheck(False)
@wraparound(False)
@cython.profile(True)
cpdef acceleration(double[:, ::1] mass, double[:, ::1] rho, long ipn, long[:, ::1] ipn_pairs, double[:, :, ::1] dw,
                   double[:, :, ::1] sig_ij, int[::1] part_type, double[:, ::1] dv):

    # Static declaration of variables.
    cdef:
        long idx, part_i, part_j
        int row, col

    # ========================================== MOMENTUM CALCULATION ==============================================
    for idx in range(1, ipn + 1):
        part_i = ipn_pairs[idx, 0]
        part_j = ipn_pairs[idx, 1]

        if part_type[part_i] == 1 or part_type[part_j] == 1:
            for row in range(3):
                for col in range(3):
                    dv[part_i, row] += mass[part_j, 0] * sig_ij[idx, row, col] * dw[idx, 0, col]
                    dv[part_j, row] += mass[part_i, 0] * sig_ij[idx, row, col] * dw[idx, 1, col]

    # ========================================== ENERGY CALCULATION ================================================
    #TODO: Implement this in the future!
# ======================================================================================================================
