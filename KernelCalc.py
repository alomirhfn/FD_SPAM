__author__ = 'alomir'

import math
import numpy as np


# ======================================================================================================================
# ======================================== Cubic Spline Kernel Class ===================================================
# ======================================================================================================================

# This class implements the cubic spline kernel, as presented in Dehnen and Aly (2012): Improving convergence in
#  smoothed particle hydrodynamics simulations without pairing instability.
class Spline:

    def __init__(self):
        pass

    @staticmethod
    def normalization_factor(h, sim_dim):

        # Kernel radius.
        r = 2 * h

        if sim_dim == 3:
            return 16 / (math.pi * math.pow(r, 3))

        elif sim_dim == 2:
            return 80 / (7 * math.pi * math.pow(r, 2))

        return 8 / (3 * r)

# ======================================================================================================================

    def kernel_calcs(self, r_ij, rij, h, sim_dim, w, dw):
        norm_factor = self.normalization_factor(h, sim_dim)

        # Kernel radius.
        r = 2 * h

        # Non-dimensional distance.
        q = rij / r

        # This is to avoid singular values for the self-contribution.
        zero_dist = np.where(rij[:] == 0)[0]
        rij[zero_dist] = 1

        indices = np.where(q[:] <= 0.5)[0]
        w[indices] = np.power(1 - q[indices], 3) - 4 * np.power(0.5 - q[indices], 3)
        dw[indices] = -(3 * np.power(1 - q[indices], 2) - 12 * np.power(0.5 - q[indices], 2)) * r_ij[indices] / \
            (rij[indices] * r)

        indices1 = np.where(q[:] > 0.5)[0]
        indices2 = np.where(q[:] < 1.0)[0]
        indices = np.intersect1d(indices1, indices2)
        w[indices] = np.power(1 - q[indices], 3)
        dw[indices] = -(3 * np.power(1 - q[indices], 2)) * r_ij[indices] / (rij[indices] * r)

        # Apply normalization factor.
        w *= norm_factor
        dw *= norm_factor

        # Restore distance.
        rij[zero_dist] = 0

        # Get rid of very small values.
        tol = 1e-9
        w[np.abs(w) <= tol] = 0
        dw[np.abs(dw) <= tol] = 0

# ======================================================================================================================

    # This method gives the value of the kernel at the particle and at a distance equal to the initial interparticle
    #  distance, both used in the artificial stress calculation.
    def calc_ref_kernel(self, wref, dp, h, sim_dim):

        r_ij_ref = np.array([np.zeros(3), [dp, 0, 0]])
        rij_ref = np.array([[0], [dp]])
        dw_ref = np.zeros((2, 3))

        self.kernel_calcs(r_ij_ref, rij_ref, h, sim_dim, wref, dw_ref)

# ======================================================================================================================
# ========================================== Wendland C6 Kernel Class ==================================================
# ======================================================================================================================


# This class implements the Wendland C6 kernel, as presented in Dehnen and Aly (2012): Improving convergence in smoothed
#  particle hydrodynamics simulations without pairing instability.
class Wendland:

    def __init__(self):
        pass

    @staticmethod
    def normalization_factor(h, sim_dim):

        # Kernel radius.
        r = 2 * h

        if sim_dim == 3:
            return 1365 / (64 * math.pi * math.pow(r, 3))

        elif sim_dim == 2:
            return 78 / (7 * math.pi * math.pow(r, 2))

        else:
            print('Wendland kernel not supported for 1D simulations!')
            print()
            return exit()

# ======================================================================================================================

    def kernel_calcs(self, r_ij, rij, h, sim_dim, w, dw):
        norm_factor = self.normalization_factor(h, sim_dim)

        # Kernel radius.
        r = 2 * h

        # Non-dimensional distance.
        q = rij / r

        # This is to avoid singular values for the self-contribution in the kernel gradient.
        zero_dist = np.where(rij[:] == 0)[0]
        rij[zero_dist] = 1

        # Non-zero kernel indices.
        indices = np.where(q[:] < 1.0)[0]

        # Auxiliary variables.
        a = 1 - q[indices]
        b = 1 + 8 * q[indices] + 25 * np.power(q[indices], 2) + 32 * np.power(q[indices], 3)
        c = 8 + 50 * q[indices] + 96 * np.power(q[indices], 2)

        # Kernel and kernel derivative evaluations.
        w[indices] = np.power(a, 8) * b
        dw[indices] = -(8 * np.power(a, 7) * b - np.power(a, 8) * c) * r_ij[indices] / (rij[indices] * r)

        # Restore distance.
        rij[zero_dist] = 0

        # Apply normalization factor.
        w *= norm_factor
        dw *= norm_factor

        # Get rid of very small values.
        tol = 1e-9
        w[np.abs(w) <= tol] = 0
        dw[np.abs(dw) <= tol] = 0

# ======================================================================================================================

    # This method gives the value of the kernel at the particle and at a distance equal to the initial inter-particle
    #  distance, both used in the artificial stress calculation.
    def calc_ref_kernel(self, wref, dp, h, sim_dim):

        r_ij_ref = np.array([np.zeros(3), [dp, 0, 0]])
        rij_ref = np.array([[0], [dp]])
        dw_ref = np.zeros((2, 3))

        self.kernel_calcs(r_ij_ref, rij_ref, h, sim_dim, wref, dw_ref)

# ======================================================================================================================
# ============================================ Quintic Spline Kernel ===================================================
# ======================================================================================================================


# This class implements the quintic spline kernel as presented in Violeau (2012), p. 315.
class QuinticSpline:

    def __init__(self):
        pass

    @staticmethod
    def normalization_factor(h, sim_dim):

        if sim_dim == 3:
            return 27 / (960 * math.pi * math.pow(h, 3))

        elif sim_dim == 2:
            return 63 / (1912 * math.pi * math.pow(h, 2))

        return 3 / (240 * h)

# ======================================================================================================================

    def kernel_calcs(self, r_ij, rij, h, sim_dim, w, dw):
        norm_factor = self.normalization_factor(h, sim_dim)

        q = rij / h

        zero = np.where(q[:] == 0)[0]
        rij[zero] = 1.0

        indices = np.where(q[:] <= 2 / 3.0)[0]

        w[indices] = np.power(3 - 3 * q[indices] / 2.0, 5) - 6 * np.power(2 - 3 * q[indices] / 2.0, 5) + 15 * \
            np.power(1 - 3 * q[indices] / 2.0, 5)

        dw[indices] = (np.power(3 - 3 * q[indices] / 2.0, 4) - 6 * np.power(2 - 3 * q[indices] / 2.0, 4) + 15 *
                       np.power(1 - 3 * q[indices] / 2.0, 4)) * r_ij[indices] / (rij[indices] * h)

        index1 = np.where(2 / 3.0 < q[:])[0]
        index2 = np.where(q[:] <= 4 / 3.0)[0]
        indices = np.intersect1d(index1, index2)

        w[indices] = np.power(3 - 3 * q[indices] / 2.0, 5) - 6 * np.power(2 - 3 * q[indices] / 2.0, 5)

        dw[indices] = (np.power(3 - 3 * q[indices] / 2.0, 4) - 6 * np.power(2 - 3 * q[indices] / 2.0, 4)) * \
            r_ij[indices] / (rij[indices] * h)

        index1 = np.where(4 / 3.0 < q[:])[0]
        index2 = np.where(q[:] < 2)[0]
        indices = np.intersect1d(index1, index2)

        w[indices] = np.power(3 - 3 * q[indices] / 2.0, 5)

        dw[indices] = np.power(3 - 3 * q[indices] / 2.0, 4) * r_ij[indices] / (rij[indices] * h)

        # Apply normalization factor.
        w *= norm_factor
        dw *= norm_factor

        # Restore distance.
        rij[zero] = 0

        # Get rid of very small values.
        tol = 1e-9
        w[np.abs(w) <= tol] = 0
        dw[np.abs(dw) <= tol] = 0

# ======================================================================================================================

    # This method gives the value of the kernel at the particle and at a distance equal to the initial interparticle
    #  distance, both used in the artificial stress calculation.
    def calc_ref_kernel(self, wref, dp, h, sim_dim):

        r_ij_ref = np.array([np.zeros(3), [dp, 0, 0]])
        rij_ref = np.array([[0], [dp]])
        dw_ref = np.zeros((2, 3))

        self.kernel_calcs(r_ij_ref, rij_ref, h, sim_dim, wref, dw_ref)

# ======================================================================================================================
