from __future__ import print_function
import numpy as np
import sys
from cython cimport boundscheck, wraparound
cimport cython

# ==================================================================================================================
# This method returns the deformation gradient tensor, which corresponds to the total deformation if the update
#  scheme is TOTAL LAGRANGIAN. Otherwise, corresponds to the incremental deformation from the updated reference
#  configuration to the current one.
#
# Where:
#  u_ij = ui_n - uj_n which is the relative displacement vector between particles i and j within the current step.
#  f = Fn+1 * Fn^-1 which is the relative deformation gradient for each particle.

@boundscheck(False)
@wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef deformation_gradient(int[::1] part_type, double[:, ::1] mass, double[:, ::1] du_ij, double[:, ::1] rho,
                           long num_parts, long[:, ::1] ipn_pairs, double[:, :, ::1] dw, double[:, :, ::1] f_tot,
                           double[:, :, ::1] f_tot_n, double[:, :, ::1] f, double[:, ::1] j, int dummy, long ipn):

    cdef:
        double[:, :, ::1] temp
        double[:, ::1] temp2
        long[:] singular
        long idx, part_i, part_j
        int row, col

    for idx in range(1, ipn + 1):
        part_i = ipn_pairs[idx, 0]
        part_j = ipn_pairs[idx, 1]

        # Deformation gradient tensor update, F_n+1 for total Lagrangian approach, or f_n+1 for updated Lagrangian.
        if part_type[part_i] == 1 or part_type[part_j] == 1:
            for row in range(3):
                for col in range(3):
                    f_tot[part_i, row, col] += mass[part_j, 0] * -du_ij[idx, row] * dw[idx, 0, col] / rho[part_j, 0]
                    f_tot[part_j, row, col] += mass[part_i, 0] * du_ij[idx, row] * dw[idx, 1, col] / rho[part_i, 0]

# ==============================================================================================================
    # Relative deformation gradient for the current step (f_n+1). This step is necessary in case of a total Lagrangian
    #  approach.
    temp = np.matmul(np.asarray(f_tot), np.linalg.inv(np.asarray(f_tot_n))) # f_n+1
    f[...] = temp[...]

# ===================================== Test for meaningful deformation field ==================================
    # Jacobian of the relative deformation gradient. If the update scheme is total Lagrangian, coincide with J.
    temp2 = np.linalg.det((np.asarray(f)))[:, None]
    j[...] = temp2[...]

    # This is to avoid singularities in case of no deformation or negative-indefinite Jacobian.
    singular = np.ascontiguousarray(np.where(np.asarray(j) <= 0)[0])

    if singular.shape[0] > 1:
        print('Negative-indefinite Jacobian at particle(s) ', np.asarray(singular))
        usr = input('Revise simulation!!! Would you like to continue? Y/N: ')
        if str(usr).upper() == "N":
            sys.exit()

# ==================================================================================================================
