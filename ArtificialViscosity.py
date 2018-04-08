__author__ = 'author'

import numpy as np
import math


class ArtificialViscosity:

    def __init__(self):
        pass

    # Classic artificial viscosity as proposed in Monagham (1982).
    @staticmethod
    def artificial_pressure(part_type, ipn_pairs, r_ij, rij, v_ij, rhoij, piij, alpha, beta, h, c):

        if alpha != 0 or beta != 0:

            # This is used to calculate the diffusion term only between fluid particles (noSlip = 'N') or between all
            #  particles (noSlip = 'Y').
            fluid_part = np.where(part_type[ipn_pairs[:, 0]] == 1)[0]

            # This is necessary so the first pair start with number 1.
            pairs = np.concatenate((np.zeros(1, dtype=int), fluid_part))

            # Used to reshape vectors.
            size = np.size(pairs)

            # Tests if particles are approaching or departing from each other.
            v_rel = np.reshape(np.sum(v_ij[pairs] * r_ij[pairs], 1), (size, 1))
            eta = 0.01 * math.pow(h, 2)

            # Artificial viscosity.
            muij = h * v_rel / (np.power(rij[pairs], 2) + eta)

            # If particles are diverging from each other the artificial pressure is zero.
            indices = np.where(v_rel[:] > 0)[0]
            muij[indices] = 0

            piij[pairs] = (alpha * c * muij - beta * np.power(muij, 2)) / rhoij[pairs]

# ======================================================================================================================
