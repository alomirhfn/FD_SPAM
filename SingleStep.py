from __future__ import print_function

__author__ = 'alomir'

# import InternalForces
import ArtificialViscosity
import Deformation
import math
import Stress
import ArtificialStress
import Hyperelasticity
import BoundaryConditions
import ConstitutiveUpdate
import numpy as np
from Timer import Timer

""" Uncomment this if you want to treat Cython files (.pyx) as Python files and see changes on the fly. """
import pyximport
import subprocess
pyximport.install()
subprocess.call(["cython", "-a", "Def_Prototype.pyx"])
subprocess.call(["cython", "-a", "Acc_Prototype.pyx"])
subprocess.call(["cython", "-a", "Dummy_Prototype.pyx"])
# subprocess.call(["cython", "-a", "Impact_Prototype.pyx"])

from Def_Prototype import deformation_gradient as def_grad
from Acc_Prototype import acceleration
from Dummy_Prototype import dummy_particles
# from Impact_Prototype import impact_force1, impact_force2


class SingleStep:

    def __init__(self):
        pass

    # This method takes care of the calculations repeated at each time step. It is here that strains, stresses,
    #  constitutive models and rates of change of energy and linear momentum are calculated as well.
    @staticmethod
    def single_step(part_type, body, mass, r, r_ij, rij, du_ij, v, v_ij, dv, rho, f_tot, f, j_tot, j, b, be, je, eps,
                    eps_e, eps_e_dev, xi, nxn, nexne, kirchhoff, tau, e, s_e, k_e, p_e, h, dp, num_parts, ipn,
                    ipn_pairs, w, dw, sim_dim, bound_parts, c, alpha, beta, bulk, shear, apsi, aphi, sy, sy0, h_mod, d,
                    eta, m, eq_time, elapsed, dummy, kernel, flag, step_size, y_criterion):

        t = Timer()

        """ This is just to get rid of PyCharm warning messages when using Cython modules. Erase it in the future! """
        r[:] *= 1.0

        # Number of significant digits for rounding up operations and tolerance for small values.
        digits = 14
        tol = math.pow(10, 1 - digits)
        np.set_printoptions(precision=16)

        # int_force_object = InternalForces.LinearMomentBalance()
        deform_object = Deformation.StrainTensors()
        diff_object = ArtificialViscosity.ArtificialViscosity()
        elast_object = Hyperelasticity.Hyperelasticity()
        stress_object = Stress.StressManipulation()
        art_stress_object = ArtificialStress.ArtificialStress()
        bc_object = BoundaryConditions.BCProperties()

        # =================================== ESPECIAL ADDITIONS COME HERE =============================================

        if elapsed < eq_time:
            gravity = body * 0.5 * (math.sin((-0.5 + elapsed / eq_time) * math.pi) + 1)
        else:
            gravity = body

        # Apply repulsive boundary forces.
        rf_flag = 0
        rf = np.zeros((num_parts + 1, 3))
        if rf_flag == 1:
            bc_object.repulsive_bc(r_ij, rij, v, v_ij, rf, dp, part_type, ipn_pairs, num_parts)

        # ==============================================================================================================
        # ==================================== VECTORS USED IN THE RATE CALCULATIONS ===================================

        # Mean inter-particle density measures.
        rhoij = 0.5 * (rho[ipn_pairs[:, 0]] + rho[ipn_pairs[:, 1]])
        rhoixj = np.reshape(rho[ipn_pairs[:, 0]] * rho[ipn_pairs[:, 1]], (ipn + 1, 1, 1))

        # Artificial stress tensor.
        piij = np.zeros((ipn + 1, 1))

        # ==============================================================================================================

        # This is used to calculate the acceleration for the first step, given an initial state of stress. This returns
        #  dv/dt_0 for the first update of velocity and position.
        if flag == 1:

            # If all stresses are zero, then, do not call acceleration.
            non_zero_sigma = np.size(np.where(kirchhoff[1:] != 0))
            if non_zero_sigma > 0:
                sij = (kirchhoff[ipn_pairs[:, 0]] + kirchhoff[ipn_pairs[:, 1]]) / rhoixj
                dv[:] = 0
                acceleration(mass, rho, ipn, ipn_pairs, dw, sij, part_type, dv)
                dv[bound_parts] = 0
                dv[np.abs(dv) < tol] = 0
                dv[:] = np.round(dv[:], decimals=digits)

            dv[:] += gravity[:] + rf[:]
            return

        # ==============================================================================================================
        # ======================================== TOTAL DEFORMATION WITHIN STEP =======================================
        # Calculate the total deformation gradient (F_n+1) and the relative deformation gradient (f_n+1) for the current
        #  time step.

        t.tic()
        f_tot_n = np.copy(f_tot)
        f_tot[:] = np.identity(3)
        def_grad(part_type, mass, du_ij, rho, num_parts, ipn_pairs, dw, f_tot, f_tot_n, f, j, dummy, ipn)
        f[bound_parts] = np.identity(3)
        f[np.abs(f) < tol] = 0
        f[:] = np.round(f[:], decimals=digits)
        j[bound_parts] = 1
        t.toc('Cython Deformation gradient')

        f_tot_t = np.transpose(f_tot, (0, 2, 1))    # Transpose of the deformation gradient.
        j_tot *= j

        # ==============================================================================================================
        # ========================================= CONSTITUTIVE CALCULATIONS ==========================================
        # Here we perform the constitutive calculations. If the material is perfectly elastic, than the trial state is
        #  the final state. If the material is viscoplastic or elastoplastic, then performs the return mapping algorithm
        #  in natural logarithm strain space.
        # TODO: Complete the description of this part of the code!

        # ========================================== CONSTITUTIVE PARAMETERS ===========================================

        # Constitutive model: EL = perfectly elastic, EP = elastoplastic or EVP = elasto-viscoplastic.
        c_model = 'EP'

        # Hyperelastic model used in the calculation of the Cauchy stress tensor:
        #  - Henky: HKY
        #  - Mooney-Rivlin: MOR
        #  - Neo-Hookean: NHK
        el_model = 'HKY'

        # ============================== CALCULATE TOTAL AND ELASTIC TRIAL STRAIN TENSORS ==============================
        # Calculate the principal elastic trial stretches, and principal directions in the natural logarithm strain
        #  space.

        # Calculate the new trial elastic state.
        t.tic()
        deform_object.strain_decomposition(f, b, be, eps, eps_e, eps_e_dev, je, nxn, nexne, num_parts, part_type, tol,
                                           digits)
        t.toc('Strain decomposition')

        # ========================================= TRIAL KIRCHHOFF STRESS TENSORS =====================================

        # Calculates the trial Cauchy stress tensor, which is the proper tensor if the constitutive model is elastic.
        ce = np.zeros((1, 3, 3))    # Elastic tangent stiffness tensor.
        elast_object.kirchhoff_stress(eps_e, eps_e_dev, je, tau, s_e, bulk, shear, ce, el_model,
                                      np.arange(num_parts + 1))

        # ==================================== CONSTITUTIVE UPLOAD (RETURN MAPPING) ====================================
        # If the material is perfectly elastic, then the trial state corresponds to final state and the calculations
        #  proceed to the next step after a few additional calculations. Otherwise, call one of the methods that apply
        #  the elastic predictor/returning map algorithm as proposed by Simo (1992), Simo and Hughes (1998), and Souza
        #  Neto et al. (2008).

        if c_model != 'EL':

            # Flag to sign if the reconstruction of the elastic left Cauchy-Green tensor is required. 0 = no
            #  reconstruction.
            reconst = np.zeros(num_parts + 1, np.int8)

            t.tic()
            const_object = ConstitutiveUpdate.ViscoPlasticity(el_model, y_criterion, c_model)
            const_object.solver(eps_e, eps_e_dev, je, xi, tau, bulk, shear, ce, aphi, apsi, sy, sy0, d, h_mod, eta, m,
                                s_e, num_parts, stress_object, elast_object, reconst, step_size, tol, digits)
            t.toc('Constitutive update')

            # ====================================== RECONSTRUCT STRAIN TENSORS ========================================

            # Particles that went through an inelastic step.
            parts = np.where(reconst == 1)
            sz = np.size(parts)

            # Vector of square principal stretches.
            lamb_e2 = np.exp(2 * eps_e[parts])

            # Reconstruct the elastic left Cauchy-Green strain tensor.
            be[parts] = np.sum(np.reshape(lamb_e2, (sz, 3, 1, 1)) * nexne[parts], 1)
            be[np.abs(be) < tol] = 0
            be[:] = np.round(be[:], decimals=digits)

        # ==============================================================================================================
        # ============================================= ARTIFICIAL VISCOSITY ===========================================

        # Artificial pressure of Monaghan (1989). It only work properly with total update Lagrangian approach.
        # t.tic()
        diff_object.artificial_pressure(part_type, ipn_pairs, r_ij, rij, v_ij, rhoij, piij, alpha, beta, h, c)
        # t.toc('Artificial viscosity')

        # ==============================================================================================================
        # ====================================== ARTIFICIAL STRESS TENSOR ==============================================

        # Artificial stress tensor to avoid tensile instability. ATTENTION: If artificial stress used, MUST TURN OFF the
        #  Shepard filter (no kernel renormalization)!
        art_stress = 0
        a = np.zeros((num_parts + 1, 3, 3))
        d = np.zeros((ipn + 1, 1))  # Artificial force scaling factor.

        if art_stress == 1:
            art_stress_object.artificialStress(r_ij, rij, rho, tau, f_tot_t, a, d, dp, h, nexne, num_parts, sim_dim,
                                               ipn, kernel)

        # Artificial forces.
        aij = np.reshape(d, (ipn + 1, 1, 1)) * (a[ipn_pairs[:, 0]] + a[ipn_pairs[:, 1]])

        # ==============================================================================================================
        # ==================================== RATE OF CHANGE OF LINEAR MOMENTUM =======================================

        # First Piola-Kirchhoff stress tensor for calculation of rate of change of linear momentum only.
        piola = np.zeros((num_parts + 1, 3, 3))

        # Reconstructing the full stress tensor in the end of the step.
        stress_object.reconstruct_tensor(kirchhoff, tau, piola, f_tot_t, j_tot, nexne, num_parts)
        piola[np.abs(piola) < tol] = 0

        # Assign stress to dummy boundary particles.
        if dummy == 1:

            # t.tic()
            piola[bound_parts] = 0
            dummy_particles(mass, rho, r_ij, piola, w, body, part_type, ipn_pairs, ipn)
            # t.toc('Dummy particles Cython')

        # Momentum per unit mass.
        sij = aij + (piola[ipn_pairs[:, 0]] + piola[ipn_pairs[:, 1]]) / rhoixj

        # Add the artificial stress combination to the diagonal terms.
        sij[:, 0, 0] += piij[:, 0]
        sij[:, 1, 1] += piij[:, 0]
        sij[:, 2, 2] += piij[:, 0]

        """ Erase this in the future. Just here to avoid warnings by PyCharm."""
        e[:] = 0
        k_e[:] = 0
        p_e[:] = 0

        # Rate of change of velocity and energy.
        # t.tic()
        dv[:] = 0.0
        acceleration(mass, rho, ipn, ipn_pairs, dw, sij, part_type, dv)
        dv[bound_parts] = 0
        dv[np.abs(dv) < tol] = 0
        dv[:] = np.round(dv[:], decimals=digits)
        dv[:] += gravity[:] + rf[:]
        # t.toc('Acceleration Cython')

        """Erase this!"""
        # Output resultant force to file. Delete all entries but the headers before starting a simulation.
        # from OutputModule import Output
        # file_path = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Moriguchi/Resultant.csv'
        # file_path2 = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Moriguchi/Resultant2.csv'
        #
        # reaction = np.zeros(3, dtype=np.float64)
        # normal = np.zeros(3)
        # normal[0] = -1.0
        #
        # if step % 10 == 0:
        #     # Model 1.
        #     impact_force1(part_type, piola, normal, dp, num_parts, reaction)
        #     if reaction.any():
        #         Output.create_impact_output(file_path, mass[1, 0] * reaction, elapsed)
        #
        #     # Model 2.
        #     wsum = np.zeros(num_parts + 1, dtype=np.float64)
        #     impact_force2(piola, w, part_type, ipn_pairs, ipn, num_parts, dp, normal, reaction, wsum)
        #     if reaction.any():
        #         Output.create_impact_output(file_path2, mass[1, 0] * reaction, elapsed)

        kirchhoff[bound_parts] = 0   # Zero boundary stresses for output.

# ======================================================================================================================
