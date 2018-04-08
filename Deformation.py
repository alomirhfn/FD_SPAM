from __future__ import print_function
__author__ = 'alomir'

import numpy as np

# from Timer import Timer


class StrainTensors:

    def __init__(self):
        pass

    # ==================================================================================================================
    # This method returns the deformation gradient tensor, which corresponds to the total deformation if the update
    #  scheme is TOTAL LAGRANGIAN. Otherwise, corresponds to the incremental deformation from the updated reference
    #  configuration to the current one.
    #
    # Where:
    #  u_ij = ui_n - uj_n which is the relative displacement vector between particles i and j within the current step.
    #  f = Fn+1 * Fn^-1 which is the relative deformation gradient for each particle.
    @staticmethod
    def deformation_gradient(part_type, mass, u_ij, rho, num_parts, ipn_pairs, dw, tot_f, f, j, dummy, tol, digits):

        # t = Timer()
        # t.tic()

        # Deformation gradient at the beginning of the current step (or at the end of the previous step - F_n).
        tot_f_n = np.copy(tot_f)

        for part in range(1, num_parts + 1):

            # For boundary particles.
            if part_type[part] == 1:

                # All pairs that contain the current particle.
                index = np.where(ipn_pairs[:, 0] == part)[0]  # Pairs (i,j).

            else:
                if dummy == 'Y':
                    index = np.array([])
                else:
                    index = np.where(ipn_pairs[:, 0] == part)[0]  # Pairs (i,j).

                    # Pairs in which the neighbor is a fluid particle.
                    fb = np.where(part_type[ipn_pairs[index, 1]] == 1)[0]
                    index = index[fb]

            size = np.size(index)

            # Deformation gradient tensor update, F_n+1 for total Lagrangian approach, or f_n+1 for updated Lagrangian.
            if size > 0:
                tot_f[part] = np.sum(np.reshape(mass[ipn_pairs[index, 1]] * -u_ij[index] / rho[ipn_pairs[index, 1]],
                                                (size, 3, 1)) * np.reshape(dw[index], (size, 1, 3)), 0) + np.identity(3)

        # ===================================== Test for meaningful deformation field ==================================

        # Jacobian of the relative deformation gradient. If the update scheme is total Lagrangian, coincide with J.
        j[:, 0] = np.linalg.det(tot_f)

        # This is to avoid singularities in case of no deformation or negative-indefinite Jacobian.
        singular = np.where(j <= 0)[0]

        if np.size(singular) > 1:
            print('Negative-indefinite Jacobian at particle(s) ', singular)
            usr = input('Revise simulation!!! Would you like to continue? Y/N: ')
            if str(usr).upper() == "N":
                exit()

        # ==============================================================================================================

        # Relative deformation gradient for the current step (f_n+1).
        f[:] = np.matmul(tot_f, np.linalg.inv(tot_f_n))  # f_n+1

        # Avoid very small numbers.
        tot_f[np.abs(tot_f) < tol] = 0
        tot_f[:] = np.round(tot_f[:], decimals=digits)
        f[np.abs(f) < tol] = 0
        f[:] = np.round(f[:], decimals=digits)

        # t.toc('Relative deformation tensors')
    # ==================================================================================================================

    # ==================================================================================================================
    # This method takes the relative deformations within the current time step and the information at its beginning to
    #  calculate the strain tensors necessary for the constitutive update.
    @staticmethod
    def strain_decomposition(f, b, be, eps, eps_e, eps_e_dev, tot_je, nxn, nexne, num_parts, part_type, tol, digits):

        # t = Timer()
        # t.tic()

        if tol < 1e-14:  # This avoids errors with the spectral decomposition.
            tol = 1e-14

        # Transpose of the relative deformation gradient.
        ft = np.transpose(f, (0, 2, 1))

        # ============================================== TOTAL STRAINS =================================================

        # Total left Cauchy-Green tensor.
        b[:] = np.matmul(np.matmul(f, b), ft)
        b[np.abs(b) < tol] = 0

        # Principal stretches (values) of b (same as the the square of those of F).
        lamb_2, n = np.linalg.eig(b)

        zro = np.where(lamb_2 <= 0)
        nnum = np.where(lamb_2 == np.nan)
        sze = np.size(zro)
        sze2 = np.size(nnum)

        if sze > 0:
            print()
            print(zro)
            print()
            lamb_2[zro] = 1

        if sze2 > 0:
            print()
            print(nnum)
            print()
            lamb_2[nnum] = 1

        lamb = np.sqrt(lamb_2)  # This is necessary hence eigvals(b) = eigvals(F)^2.

        # This is to rearrange ne because the eig function puts the vectors in columns instead of rows.
        n[:] = np.transpose(n, (0, 2, 1))

        # ============================================ ELASTIC STRAINS =================================================

        # Elastic left Cauchy-Green tensor.
        be[:] = np.matmul(np.matmul(f, be), ft)
        be[np.abs(be) < tol] = 0

        # Principal stretches (values) and vectors of be. The latter are the same as those of Fe.
        lamb_e2, ne = np.linalg.eig(be)

        zro = np.where(lamb_e2 <= 0)
        nnum = np.where(lamb_e2 == np.nan)
        sze = np.size(zro)
        sze2 = np.size(nnum)

        if sze > 0:
            print()
            print(zro)
            print()
            lamb_e2[zro] = 1

        if sze2 > 0:
            print()
            print(nnum)
            print()
            lamb_e2[nnum] = 1

        lamb_e = np.sqrt(lamb_e2)  # This is necessary hence eigvals(be) = eigvals(Fe)^2.

        # This is to rearrange ne because the eig function puts the vectors in columns instead of rows.
        ne[:] = np.transpose(ne, (0, 2, 1))

        # ========================= Sorting the eigenvalues and vectors in ascending order =============================

        # Arguments to sort the eigenvalues and eigenvectors of the total strain.
        args = np.argsort(lamb, 1)
        copy_lamb = np.copy(lamb)
        copy_n = np.copy(n)

        # Arguments to sort the eigenvalues and eigenvectors of the elastic strain.
        args2 = np.argsort(lamb_e, 1)
        copy_lambe = np.copy(lamb_e)
        copy_ne = np.copy(ne)

        for part in range(1, num_parts + 1):

            if part_type[part] == 1:

                lamb[part, :] = copy_lamb[part, args[part]]
                n[part, :] = copy_n[part, args[part]]

                lamb_e[part, :] = copy_lambe[part, args2[part]]
                ne[part, :] = copy_ne[part, args2[part]]
        # ==============================================================================================================

        # ============================================== TOTAL STRAINS =================================================

        # This is required to calculate the outer product of the principal directions.
        nr = np.reshape(n, (3 * (num_parts + 1), 3, 1))
        nrt = np.reshape(n, (3 * (num_parts + 1), 1, 3))
        nxn[:] = np.reshape(nr * nrt, (num_parts + 1, 3, 3, 3))

        # Principal total logarithmic stretches (logarithmic eigenvalues of F).
        eps[:] = np.log(1e16 * lamb) - np.log(1e16)     # This avoids errors due to stretches near 1.
        eps[np.abs(eps) < tol] = 0

        # ============================================ ELASTIC STRAINS =================================================

        # This is required to calculate the outer product of the principal directions.
        ner = np.reshape(ne, (3 * (num_parts + 1), 3, 1))
        nert = np.reshape(ne, (3 * (num_parts + 1), 1, 3))
        nexne[:] = np.reshape(ner * nert, (num_parts + 1, 3, 3, 3))

        # Trace of the the trial elastic deformation gradient tensor (square root of the determinant of be_tr).
        tot_je[:, 0] = lamb_e[:, 0] * lamb_e[:, 1] * lamb_e[:, 2]

        # Principal trial elastic logarithmic stretches (logarithmic eigenvalues of Fe).
        eps_e[:] = np.log(1e16 * lamb_e) - np.log(1e16)
        eps_e[np.abs(eps_e) < tol] = 0

        # Deviatoric part of eps_e_tr.
        eps_e_dev[:] = eps_e - np.sum(eps_e, 1, keepdims=True) / 3
        eps_e_dev[np.abs(eps_e_dev) < tol] = 0

        # ========================================= ROUND UP OPERATIONS ================================================

        b[:] = np.round(b[:], decimals=digits)
        be[:] = np.round(be[:], decimals=digits)
        eps[:] = np.round(eps[:], decimals=digits)
        eps_e[:] = np.round(eps_e[:], decimals=digits)
        eps_e_dev[:] = np.round(eps_e_dev[:], decimals=digits)

        # t.toc('Trial tensors')

# ======================================================================================================================
