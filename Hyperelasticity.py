__author__ = 'alomir'

import numpy as np


class Hyperelasticity:

    def __init__(self):
        pass

    # ==================================================================================================================
    # This method returns the Kirchhoff stress tensor based on hyperelastic models. Note that the stress tensor is
    #  returned in Voight notation (a 3x1 vector of principal stresses). In the method, model refers to one of the
    #  following:
    #
    #   - Modified Saint-Venant-Kirchhoff: SVK (as in Hughes and Simo, 1998)
    #   - Ogden: OGD
    #   - Mooney-Rivlin: MOR
    #   - Neo-Hookean: NHK
    #
    #  The other models are all described in detail in the book of Holzapfel (2010).
    def kirchhoff_stress(self, eps, eps_dev, je, tau, s_e, bulk, shear, ce, model, parts):

        # Henky logarithmic model.
        if model == "HKY":

            ln_je = np.sum(eps[parts], 1, keepdims=True)      # This avoids errors with je close to 1.

            # Stored elastic energy.
            s_e[parts] = bulk * np.power(ln_je, 2) / 2 + shear * np.sum(eps_dev[parts] * eps_dev[parts], 1,
                                                                        keepdims=True)

            # Kirchhoff stress tensor. The derivative of the stored energy function with respect to lambda.
            tau[parts] = bulk * ln_je + 2 * shear * eps_dev[parts]

            # The elastic tangent modulus.
            ce[0] = bulk * np.ones((3, 3)) + 2 * shear * (np.identity(3) - np.ones((3, 3)) / 3)

        # Generalized Ogden model as presented in Souza Neto et al. (2008).
        else:

            # Principal stretches.
            lt = np.exp(eps[parts])

            # Principal deviatoric stretches.
            ldev = np.exp(eps_dev[parts])

            if model == "MOR":
                mu1 = 0.875 * shear
                mu2 = -0.125 * shear
                a1 = 2
                a2 = -2

            elif model == "NHK":
                mu1 = shear
                mu2 = 0
                a1 = 2
                a2 = 0

            else:
                print('Invalid input: this model is not implemented!')
                print('Set to default model - Modified Saint-Venant - Kirchhoff (Henky model)')
                self.kirchhoff_stress(eps, eps_dev, je, tau, s_e, bulk, shear, ce, "SVK", parts)
                return

            # For stored energy.
            se1 = mu1 * (np.sum(np.power(ldev, a1), 1, keepdims=True) - 3) / a1

            if a2 != 0:
                se2 = mu2 * (np.sum(np.power(ldev, a2), 1, keepdims=True) - 3) / a2
            else:
                se2 = 0

            # For stress.
            t1 = mu1 * np.power(je[parts], -a1 / 3) * (np.power(lt, a1) - np.sum(np.power(lt, a1), 1, keepdims=True) /
                                                       3)
            t2 = mu2 * np.power(je[parts], -a2 / 3) * (np.power(lt, a2) - np.sum(np.power(lt, a2), 1, keepdims=True) /
                                                       3)
            # T3, T4 .... can add new terms for different models.

            # Stored elastic energy.
            s_e[parts] = bulk * ((np.power(je[parts], 3) / 3) - 0.5 * np.power(je[parts], 2)) + se1 + se2  # + se3 + ...

            # Kirchhoff stress tensor. The derivative of the stored energy function with respect to lambda.
            tau[parts] = bulk * (np.power(je[parts], 2) - je[parts]) + t1 + t2  # + T3 + ...

# ======================================================================================================================
