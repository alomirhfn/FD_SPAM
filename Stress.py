__author__ = 'alomir'

import numpy as np


class StressManipulation:

    def __init__(self):
        pass

# ======================================================================================================================
    # Auxiliary method to calculate the norm of a vector, more specifically, the norm of the deviatoric part of a tensor
    #  in Voight notation (vector).
    @staticmethod
    def calc_j2(a, j2, parts):

        j2[parts] = np.sum(a[parts] * a[parts], 1, keepdims=True) / 2

# ======================================================================================================================
    # This method decompose the stress tensor into the hydrostatic (spheric) part and a deviatoric part. Use when the
    #  stress tensor is in Voight notation in principal representation (vector).
    @staticmethod
    def stress_tensor_decomposition(tensor, dev, p, parts):

        # Hydrostatic part of the stress tensor.
        p[parts] = np.sum(tensor[parts], 1, keepdims=True) / 3

        # Deviatoric part of the stress tensor.
        dev[parts] = tensor[parts] - p[parts]

# ======================================================================================================================
    # This method determines the Cauchy stress tensor based on the principal directions and values of Tau.
    @staticmethod
    def reconstruct_tensor(kirchhoff, tau, piola, ft, j, nxn, num_parts):

        # Cauchy stress tensor.
        kirchhoff[:] = np.sum(np.reshape(tau, (num_parts + 1, 3, 1, 1)) * nxn, 1)

        # First piola-Kirchhoff stress tensor.
        piola[:] = np.matmul(kirchhoff, np.linalg.inv(ft))

        # Second Piola-Kirchhoff stress tensor.
        # SPiola = np.matmul(np.linalg.inv(F),piola)

# ======================================================================================================================
