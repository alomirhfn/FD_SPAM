# This module implements the solid wall boundary conditions as proposed by Adami et al. (2012) in the paper entitled:
#  "A generalized wall boundary condition for smoothed particle hydrodynamics".

__author__ = 'alomir'

import numpy as np


class BCProperties:

    def __init__(self):
        pass

# ======================================================================================================================
    # This function implements the periodic boundary conditions as developed by myself.
    @staticmethod
    def periodic_bc(r, r_ij, x_min, x_max, y_min, y_max, z_min, z_max, dp, h, pbcx, pbcy, pbcz, ipn_pairs, flag):

        # Kernel radius.
        kr = 2 * h

        # This part reposition the particles that are displaced outside the domain. It is only necessary to call when
        #  updating particles' positions. Not necessary when performing NNPS algorithms.
        if flag == 1:
            if pbcx == 1:
                out_x_plus = np.where(r[:, 0] > x_max + dp / 2)[0]
                out_x_minus = np.where(r[:, 0] < x_min - dp / 2)[0]
                r[out_x_plus, 0] += -x_max + x_min - dp
                r[out_x_minus, 0] += x_max - x_min + dp

            if pbcy == 1:
                out_y_plus = np.where(r[:, 1] > y_max + dp / 2)[0]
                out_y_minus = np.where(r[:, 1] < y_min - dp / 2)[0]
                r[out_y_plus, 1] += -y_max + y_min - dp
                r[out_y_minus, 1] += y_max - y_min + dp

            if pbcz == 1:
                out_z_plus = np.where(r[:, 2] > z_max + dp / 2)[0]
                out_z_minus = np.where(r[:, 2] < z_min - dp / 2)[0]
                r[out_z_plus, 2] += -z_max + z_min - dp
                r[out_z_minus, 2] += z_max - z_min + dp

            # Recalculate the pair-wise distance vector.
            r_ij[:] = r[ipn_pairs[:, 0]] - r[ipn_pairs[:, 1]]

        # This part corrects the pair-wise distance vector to account for periodicity.
        if pbcx == 1:
            x_dist = np.abs(r_ij[:, 0])
            periodic_xdist = x_dist - x_max + x_min - dp
            more = np.where(np.abs(periodic_xdist) >= x_dist)[0]
            periodic_xdist[more] = x_dist[more]
            neighbors = np.where(np.abs(periodic_xdist) <= kr)[0]
            r_ij[neighbors, 0] = periodic_xdist[neighbors] * np.sign(r_ij[neighbors, 0])

        if pbcy == 1:
            y_dist = np.abs(r_ij[:, 1])
            periodic_ydist = y_dist - y_max + y_min - dp
            more = np.where(np.abs(periodic_ydist) >= y_dist)[0]
            periodic_ydist[more] = y_dist[more]
            neighbors = np.where(np.abs(periodic_ydist) <= kr)[0]
            r_ij[neighbors, 1] = periodic_ydist[neighbors] * np.sign(r_ij[neighbors, 1])

        if pbcz == 1:
            z_dist = np.abs(r_ij[:, 2])
            periodic_zdist = z_dist - z_max + z_min - dp
            more = np.where(np.abs(periodic_zdist) >= z_dist)[0]
            periodic_zdist[more] = z_dist[more]
            neighbors = np.where(np.abs(periodic_zdist) <= kr)[0]
            r_ij[neighbors, 2] = periodic_zdist[neighbors] * np.sign(r_ij[neighbors, 2])

# ======================================================================================================================
    # This function implements the Monaghan type repulsive boundary conditions in its original form. This repulsive
    #  force consider that the wall is fixed (vw = 0).
    @staticmethod
    def repulsive_bc(r_ij, rij, v, v_ij, rf, dp, part_type, ipn_pairs, num_parts):

        # Fine tuning parameters.
        p1 = 4
        p2 = 2
        a = 0.2

        # Repulsive force scaling factor.
        df = a * np.sum(v * v, 1, keepdims=True)

        # Interaction distance ratio.
        zero = np.where(rij == 0)[0]
        rij[zero] = 1
        f = dp / rij
        rij[zero] = 0

        # Pairs that are not within interaction range.
        far = np.where(f < 1)[0]
        f[far] = 0

        # Comment these next 3 parts of code to enforce tensile boundary forces as well.
        # ===============================================================================

        # Check if particle is moving towards the boundary.
        appr = np.sum(r_ij * v_ij, 1)

        # Particles that are moving away from the wall.
        away = np.where(appr > 0)[0]

        # Set them to zero (no tensile forces).
        f[away] = 0
        # ===============================================================================

        for part in range(1, num_parts + 1):

            if part_type[part] == 1:

                # All pairs that contain the fluid particle.
                index = np.where(ipn_pairs[:, 0] == part)[0]

                # The indices in the previous arrays where the second particle is a repulsive boundary particle, type 3.
                bp = np.where(part_type[ipn_pairs[index, 1]] == 3)[0]

                # Select only the pairs that have the particle and a boundary one.
                index = index[bp]

                # Repulsive force of Lennard-Jones part_type as presented in Monaghan (1994).
                rf[part] = df[part] * np.sum((np.power(f[index], p1) - np.power(f[index], p2)) * r_ij[index] /
                                             np.power(rij[index], 2), 0)

        rf[np.abs(rf) <= 1e-12] = 0   # To avoid very small numbers due to truncation.

# ======================================================================================================================
    # This method calculates the stresses on the boundary particles which influence the velocity and displacement of the
    #  particles. Similarly to Adami et al. (2012) here, an averaged value of the neighborhood of fluid particles is
    #  assigned to each dummy particle.
    @staticmethod
    def dummy_particles(mass, rho, r_ij, piola, w, body, part_type, ipn_pairs):

        # Pairs with one boundary and one fluid particle.
        d = np.where(part_type[ipn_pairs[:, 0]] != 1)[0]
        f = np.where(part_type[ipn_pairs[:, 1]] == 1)[0]
        p = np.intersect1d(d, f)
        dummy = np.unique(ipn_pairs[p, 0])

        for bpart in dummy:

            # All pairs which contain the current boundary particle.
            index = np.where(ipn_pairs[:, 0] == bpart)[0]

            # Only pairs where the interacting particle is a fluid particle.
            fluid = np.where(part_type[ipn_pairs[index, 1]] == 1)[0]
            index = index[fluid]
            size = np.size(index)

            wsum = np.sum((mass[ipn_pairs[index, 1]] / rho[ipn_pairs[index, 1]]) * w[index], keepdims=True)

            # This is to avoid divisions by zero when the boundary particle has no fluid neighbor.
            if wsum > 0:

                temp_piola = np.sum((piola[ipn_pairs[index, 1]] + np.reshape(rho[ipn_pairs[index, 1]] * body[
                    ipn_pairs[index, 1]], (size, 3, 1)) * np.reshape(-r_ij[index], (size, 1, 3))) *
                                    np.reshape((mass[ipn_pairs[index, 1]] / rho[ipn_pairs[index, 1]]) *
                                               w[index], (size, 1, 1)), 0)

                piola[bpart] = temp_piola / wsum

            else:
                piola[bpart, :] = 0

# ======================================================================================================================
