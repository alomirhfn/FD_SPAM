__author__ = 'alomir'

import numpy as np
# from Timer import Timer


class Filters:

    def __init__(self):
        pass

# ======================================================================================================================
    # This method renormalizes the kernel so that it restores zero-th and first order consistency. The formulation was
    #  taken from: Variational and momentum preservation aspects of Smooth Particle Hydrodynamic formulations - Bonet
    #  and Lok (1999).
    @staticmethod
    def kernel_renorm(part_type, mass, rho, r_ij, ipn_pairs, w, num_parts, sim_dim, n_order):

        # t = Timer()
        # t.tic()

        # 0 = No normalization.
        if n_order == 0:
            return

        for part in range(1, num_parts + 1):

            if part_type[part] == 1:

                # Pairs (i,j).
                index = np.where(ipn_pairs[:, 0] == part)[0]
                size = np.size(index)

                wc_sum = np.sum((mass[ipn_pairs[index, 1]] / rho[ipn_pairs[index, 1]]) * w[index], 0)

                if n_order == 1:

                    c = 1 / wc_sum

                else:

                    wl_sum = np.sum((mass[ipn_pairs[index, 1]] * w[index] / rho[ipn_pairs[index, 1]]) *
                                    -r_ij[index], 0, keepdims=True)

                    d = np.reshape(r_ij[index], (size, 3, 1)) * np.reshape(r_ij[index], (size, 1, 3))

                    wt_sum = np.sum(np.reshape(mass[ipn_pairs[index, 1]] * w[index] /
                                               rho[ipn_pairs[index, 1]], (size, 1, 1)) * d, 0)

                    # This is so that for sim_dim less than 3, the matrix WTsum is not singular.
                    if sim_dim == 1:
                        wt_sum[1, 1] = wt_sum[2, 2] = 1
                    elif sim_dim == 2:
                        wt_sum[2, 2] = 1

                    # Correction vector as proposed in Bonet and Lok (1999).
                    if np.linalg.det(wt_sum) > 0:
                        b = -np.dot(wl_sum, np.linalg.inv(wt_sum))
                    else:
                        b = -wl_sum

                    # Renormalization factor as proposed in Bonet and Lok (1999).
                    a = 1 / (wc_sum + np.dot(b, np.reshape(wl_sum, (3, 1))))

                    # Normalization factor.
                    c = a * (1 + np.dot(b, np.reshape(-r_ij[index], (size, 3, 1))))

                # Zero-th and first order renormalized kernel.
                w[index, 0] = w[index, 0] * c

        # t.toc('Kernel renormalization')

# ======================================================================================================================
    # This method restores 0th and 1st order consistency of the kernel derivative. This formulation was derived by
    #  myself, however it is based on Bonet and Lok (1999) and Violeau (2012).
    @staticmethod
    def kernel_deriv_correct(part_type, mass, rho, r_ij, ipn_pairs, w, dw, num_parts, sim_dim, cn_option, n_order):

        # t = Timer()
        # t.tic()

        # Tolerance for zero.
        tol = 1e-16

        # Renormalization of the kernel.
        Filters.kernel_renorm(part_type, mass, rho, r_ij, ipn_pairs, w, num_parts, sim_dim, n_order)

        if cn_option == 0:
            return

        # Calculate the matrices M.
        for part in range(1, num_parts + 1):

            if part_type[part] == 1:

                # This is the matrix M.
                m = np.identity(3, dtype=np.float64)

                # The matrix L is the inverse of matrix M.
                g = np.identity(3, dtype=np.float64)

                # Initializing the vector of sums of the kernel gradient.
                d_gamma = np.zeros((1, 3))

                # Pairs (i,j).
                index = np.where(ipn_pairs[:, 0] == part)[0]
                size = np.size(index)

                # The matrix M basically represent the sum of the kernel derivatives of particle i if the problem is 1D.
                m[0:sim_dim, 0:sim_dim] = (np.sum(np.reshape(mass[ipn_pairs[index, 1]] * -r_ij[index] /
                                                             rho[ipn_pairs[index, 1]], (size, 3, 1)) *
                                                  np.reshape(dw[index], (size, 1, 3)), 0))[0:sim_dim, 0:sim_dim]
                # The last part in square brackets is in case of a 2D problem, the z-component is 1 and det(M) != 0.

                # Eliminate small residues.
                m[np.abs(m) <= tol] = 0

                if sim_dim == 1:
                    if np.abs(m[0, 0]) > 0:
                        g[0, 0] = 1.0 / m[0, 0]
                else:
                    if np.linalg.det(m) != 0:
                        g = np.linalg.inv(m)

                # Avoid small truncation errors.
                g[np.abs(g) <= tol] = 0

                # 0th- and 1st-order corrected kernel gradient.
                if cn_option == 2:

                    # Sum of the gradient of the kernel.
                    d_gamma = np.sum((mass[ipn_pairs[index, 1]] /
                                      rho[ipn_pairs[index, 1]]) * dw[index], 0, keepdims=True)

                    d_gamma[np.abs(d_gamma) <= tol] = 0   # Avoid small numbers.

                # Kernel gradient correction as presented in Bonet and Lok (1999).
                dw[index] = (dw[index] - w[index] * d_gamma) * np.mat(g)

        # t.toc('Kernel derivative correction')
# ======================================================================================================================
