__author__ = 'alomir'

import numpy as np

class XSPH:

    def __init__(self):
        pass

    #===================================================================================================================
    # This method implements the XSPH model as presented in Violeau (2012), p.455. According to this reference, there is
    #  an introduction of viscosity, but linear momentum is preserved. It also has a benefit in terms of stability of
    #  the method. Recommended values of epsilon are 0.5 <= epsilon <= 0.65 (Violeau recommends 0.55).
    def calcVelMod(self, part_type, xVel, mass, rho, v_ij, W, epsilon, IPNPairs, numParts):

        rhoij = (rho[IPNPairs[:,0]] + rho[IPNPairs[:,1]])

        for part in range(1,numParts + 1):

            if(part_type[part] == 1):

                index1 = np.where(IPNPairs[:,0] == part)[0]
                index2 = np.where(IPNPairs[:,1] == part)[0]

                # ===== Only interactions with fluid particles ===========
                fluid1 = np.where(part_type[IPNPairs[index1, 1]] == 1)[0]
                fluid2 = np.where(part_type[IPNPairs[index2, 0]] == 1)[0]
                index1 = index1[fluid1]
                index2 = index2[fluid2]
                # ========================================================

                Wsum = np.sum((mass[IPNPairs[index1, 1]] / rho[IPNPairs[index1, 1]]) * W[index1, 0], keepdims=True) + \
                       np.sum((mass[IPNPairs[index2, 0]] / rho[IPNPairs[index2, 0]]) * W[index2, 1], keepdims=True)

                xVel[part] = (np.sum((mass[IPNPairs[index1,1]] * W[index1,0] / rhoij[index1]) * -v_ij[index1],0) +
                             np.sum((mass[IPNPairs[index2,0]] * W[index2,1] / rhoij[index2]) * v_ij[index2],0)) / Wsum

        xVel *=  2 * epsilon    # The factor 2 comes from rhoij = (rhoi + rhoj) / 2
#=======================================================================================================================