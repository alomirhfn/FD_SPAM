__author__ = 'alomir'

import numpy as np

# from Timer import Timer


class LinearMomentBalance:

    def __init__(self):
        pass

# ======================================================================================================================
    # This method calculates the rates of change of momentum and energy.
    @staticmethod
    def acceleration(mass, rho, r, v, ipn_pairs, dw, num_parts, sig_ij, part_type, dv, e, s_e, k_e, p_e, j, gravity, rf,
                     tol, digits):

        # ========================================== MOMENTUM CALCULATION ==============================================

        # Numpy precision of print statements.
        # np.set_printoptions(precision=32)

        # t = Timer()
        # t.tic()
        for part in range(1, num_parts + 1):

            if part_type[part] == 1:

                # Returns all pairs that contain the particles under consideration.
                index = np.where(ipn_pairs[:, 0] == part)[0]
                size = np.size(index)

                # Total momentum due to the the stress tensor times unit volume.
                m_sij = np.reshape(mass[ipn_pairs[index, 1]], (size, 1, 1)) * sig_ij[index]

                # Rate of change of linear momentum for particle "i".
                dv_sig1 = m_sij * np.reshape(np.repeat(dw[index], 3, 0), (size, 3, 3))

                dv[part] = np.sum(np.sum(dv_sig1, 2), 0)

        # t.toc('Acceleration loop')

        # t.tic()
        # Get rid of small accelerations and rounds up them to the number of significant digits.
        dv[np.abs(dv) < tol] = 0
        dv[:] = np.round(dv[:], decimals=digits)

        # Add body forces.
        dv[:] += gravity[:] + rf[:]
        # t.toc('Linear momentum')

        # ========================================== ENERGY CALCULATION ================================================
        # # t.tic()
        # # Increment of total mechanical elastic energy. This is not a rate!
        # s_e[:] *= mass / (j * rho)
        # s_e[np.abs(s_e) <= tol] = 0
        #
        # # Kinetic energy.
        # k_e[:] = 0.5 * mass * np.sum(v * v, 1, keepdims=True)
        # k_e[np.abs(k_e) <= tol] = 0
        #
        # # Positional potential energy.
        # p_e[:] = mass * np.sum(r * gravity, 1, keepdims=True)
        #
        # # Total particle energy.
        # e[:] = k_e + s_e + p_e
        # # t.toc('Energy')
# ======================================================================================================================
