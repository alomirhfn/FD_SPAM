from __future__ import print_function

__author__ = 'alomir'

# import NNPS
# from KernelCalc import Wendland, Spline, QuinticSpline
# import Filters
import SimParameters
import SingleStep
import BoundaryConditions
import XSPH
import numpy as np
from Timer import Timer

""" Uncomment this if you want to treat Cython files (.pyx) as Python files and see changes on the fly. """
import pyximport
import subprocess
pyximport.install()
# subprocess.call(["cython", "-a", "Kernel_Prototype.pyx"])
subprocess.call(["cython", "-a", "NNPS_Prototype.pyx"])
subprocess.call(["cython", "-a", "Corrections_Prototype.pyx"])

from NNPS_Prototype import direct_search, create_grid, allocate_particles, linked_list
from Corrections_Prototype import kernel_sum, shepard_filter, kernel_deriv_moments, kernel_grad_correction
from Kernel_Prototype import kernel_calcs


class Solver:

    def __init__(self, params_list, init_r):
        self.params = SimParameters.Initialization(params_list, init_r)

    def solve(self, part_type, r0, v0, rho0, mass, sigma0, e0, body, output_object):

        # Number of significant digits for rounding up operations and tolerance for small values.
        digits = 14
        tol = 10**(1 - digits)
        np.set_printoptions(precision=16)

        # ==============================================================================================================
        # ===================================== PARTICLES INITIALIZATION ===============================================

        step_object = SingleStep.SingleStep()
        xsph_object = XSPH.XSPH()
        bc_object = BoundaryConditions.BCProperties()
        t = Timer()

        # ==============================================================================================================
        # ====================================== READING INPUT FROM USER ===============================================

        dp = self.params.get_init_part_dist()
        h = self.params.get_h()
        step_min = self.params.get_dt0()
        num_steps = self.params.get_num_steps()
        kernel = self.params.kernel_option()
        sim_dim = self.params.get_sim_dim()
        c_n_option = self.params.kernel_correction_option()
        n_order = self.params.kernel_correction_order()
        nnps = self.params.nnps_option()
        xsph = self.params.xsph_option()
        integration = self.params.integration_scheme_option()
        alpha = self.params.get_artvisc_alpha()
        beta = self.params.get_artvisc_beta()
        c = self.params.get_sound_speed()
        epsilon = self.params.get_xsph_epsilon()
        cohesion, phi = self.params.get_mohr_coulomb_params()
        mu0 = self.params.get_dynamic_visc()
        x_div, y_div, z_div = self.params.get_grid_divisions()
        pbc_option = self.params.get_pbc_option()
        pbc_x, pbc_y, pbc_z = self.params.get_pbc_directions()
        dummy = self.params.dummy_particles_option()
        eq_time = self.params.get_equilibration_time()
        a_psi, a_phi, kc = self.params.get_drucker_prager_params()
        young, nu, bulk, shear = self.params.get_elastic_params()
        dx_min, dx_max, dy_min, dy_max, dz_min, dz_max = self.params.get_domain()
        x_min, x_max, y_min, y_max, z_min, z_max = self.params.get_pbc_domain()
        num_cells = self.params.get_num_cells()
        num_parts = np.size(r0, 0) - 1
        step_size = step_min

        # ==============================================================================================================
        # ========================================= INITIALIZING DATA ARRAYS ===========================================

        r = np.copy(r0)             # Position
        v = np.copy(v0)             # Velocity
        rho = np.copy(rho0)         # Density
        e = np.copy(e0)             # Energy
        kirchhoff = np.copy(sigma0)     # Cauchy stress tensor

        u = np.zeros((num_parts + 1, 3))              # Updated displacement of the particle.
        dv = np.zeros((num_parts + 1, 3))             # Rate of change of linear momentum.
        x_vel = np.zeros((num_parts + 1, 3))          # Modified velocity (Monagham).
        s_e = np.zeros((num_parts + 1, 1))            # Internal elastic stored energy.
        k_e = np.zeros((num_parts + 1, 1))            # Kinetic energy.
        p_e = np.zeros((num_parts + 1, 1))            # Potential energy.
        xi = np.zeros((num_parts + 1, 1))             # Cumulative plastic strain-like internal variable for hardening.
        b = np.zeros((num_parts + 1, 3, 3))           # Total left Cauchy-Green deformation tensor.
        be = np.zeros((num_parts + 1, 3, 3))          # Elastic left Cauchy-Green deformation tensor.
        f_tot = np.zeros((num_parts + 1, 3, 3))       # Total deformation gradient tensor.
        f = np.zeros((num_parts + 1, 3, 3))           # Relative deformation gradient tensor.
        j_tot = np.ones((num_parts + 1, 1))           # Jacobian of the total deformation gradient (det[F]).
        j = np.ones((num_parts + 1, 1))               # Jacobian of the relative deformation gradient (det[f]).
        eps = np.zeros((num_parts + 1, 3))            # Principal logarithmic stretches.
        eps_e = np.zeros((num_parts + 1, 3))          # Principal elastic logarithmic stretches.
        eps_e_dev = np.zeros((num_parts + 1, 3))      # Principal elastic deviatoric logarithmic stretches.
        je = np.ones((num_parts + 1, 1))              # Jacobian of the elastic stretch tensor (det[Fe]).
        nxn = np.zeros((num_parts + 1, 3, 3, 3))      # Principal directions in the reference cartesian axes.
        nexne = np.zeros((num_parts + 1, 3, 3, 3))    # Principal elastic directions in the reference cartesian axes.
        tau = np.zeros((num_parts + 1, 3))            # Kirchhoff stress tensor.
        sy = np.zeros((num_parts + 1, 1))             # Yield surface radius.

        # For undeformed reference configuration:
        b[:] = be[:] = f[:] = f_tot[:] = np.identity(3)

        # Velocity at half step used only in the Leap-frog integration scheme.
        v_half = np.copy(v)

        # Previous particle displacement.
        un = np.copy(u)

        # ==============================================================================================================
        # ==================================== INITIALIZING SIMULATION TIME ============================================

        sim_time = num_steps * step_min       # Total time of simulation.
        elapsed = 0.0         # Elapsed time of the simulation.
        step = 0

        # ==============================================================================================================
        # =========================================== KERNEL CHOICE ====================================================

        # Choice of Kernel.
        if kernel == 'S':
            # kernel_object = Spline()
            kernel_type = 1
        elif kernel == 'QS':
            kernel_type = 3
            # kernel_object = QuinticSpline()
        else:
            if sim_dim == 1:
                print('Wendland kernel not supported for 1D simulations. Choose another kernel!')
                exit()
            kernel_type = 2
            # kernel_object = Wendland()

        # ==============================================================================================================
        # ======================================= ADDITIONAL INPUT, INSERT  HERE =======================================

        # Output degree of information, F = all information and S = simplified.
        version = 'F'

        # Yielding criterion: EL = Perfect elasticity, VM = Von Mises (Perfect plasticity), DP = Drucker-Prager,
        # MC = Mohr-Coulomb, or MCC = Modified Cam-Clay.
        y_criterion = 'VM'

        # Initial yield stress.
        sy[:] = cohesion
        if y_criterion == 'DP' and a_phi > 1e-3:
            sy *= kc
        sy0 = np.copy(sy)

        # Plastic hardening modulus. Set zero for perfect plasticity.
        tang_mod = 0
        hard_mod = tang_mod / (1 - tang_mod / young)
        gen_mod = hard_mod * np.ones((num_parts + 1, 1))   # Generalized hardening modulus.

        # Kinematic viscosity for viscoplastic response.
        if cohesion > 0:
            dyn_visc = mu0 / cohesion     # Pa.s (should be in the range of 0.001 to 1 young modulus).
        else:
            dyn_visc = mu0
        exm = 1    # Exponent of the Pierce model of viscoplasticity. Set m = 1 for linear model (Bingham).

        # General division of particles.
        fluid = np.where(part_type[:] == 1)[0]
        bound_parts = np.where(part_type[:] != 1)[0]

        # ==============================================================================================================
        # ==============================================================================================================

        # Output all initial information.
        print('Printing initial conditions...')
        file_num = 0   # Sequential number of output files.
        output_object.create_csv_file(r, u, v, rho, s_e, k_e, kirchhoff, eps, eps_e, num_parts, nxn, nexne, 0, version)
        print('Number of steps =', num_steps)
        print()

    # ==================================================================================================================
    # =========================================== TOTAL LAGRANGEAN KERNEL ==============================================
    # ==================================================================================================================

        # Here I calculate the nearest neighbor list, the consistency corrections, kernel and kernel derivatives with
        #  respect to the reference configuration for use with total Lagrange approach or updated Lagrangean.

        # ==============================================================================================================
        # ================================== NNPS ALGORITHM AND KERNEL CALCULATION =====================================

        # Initial conditions for the nearest particle searching algorithm using Cython.
        ipn0 = trial_size = int(min(sim_dim * 10 ** 6, num_parts * (4 * h / dp) ** sim_dim))
        ipn_pairs0 = -np.ones((trial_size, 2), dtype=int)
        undersized = 1
        grid = None

        # Linked-list nearest neighbor search.
        if nnps.upper() == 'LL':
            grid = np.zeros((num_cells + 1, 3**sim_dim + (2**sim_dim) * (pbc_x + pbc_y + pbc_z)), dtype=int)
            cells_parts = np.zeros((num_cells + 1, num_parts + 1), dtype=int)
            parts_cells = np.zeros(num_parts + 1, dtype=int)

            # Create an imaginary mesh of the entire domain with fixed size length equal to 2h and return an array with
            #  all cells that are immidate neighbors of any given cell.
            create_grid(x_div, y_div, z_div, grid, pbc_option, pbc_x, pbc_y, pbc_z, x_max, y_max, z_max, dp, h)

            # Map the particles to their respective cells and vice-versa.
            max_index = allocate_particles(r0, x_div, y_div, num_cells, h, num_parts, cells_parts, parts_cells, dp)
            cells_parts = np.ascontiguousarray(cells_parts[:, 1:max_index])   # Gets rid of the first column.

            # t.tic()
            while undersized == 1:
                ipn_pairs0 = -np.ones((trial_size, 2), dtype=int)
                ipn0, undersized = linked_list(r0, parts_cells, cells_parts, grid, h, dp, sim_dim, num_parts,
                                               ipn_pairs0, trial_size, x_min, x_max, y_min, y_max, z_min, z_max,
                                               pbc_option, pbc_x, pbc_y, pbc_z)
                trial_size += int(num_parts * 10)
            # t.toc('Linked-List Algorithm Cython')

        # Direct nearest neighbor search.
        else:
            # t.tic()
            while undersized == 1:
                ipn_pairs0 = -np.ones((trial_size, 2), dtype=int)
                ipn0, undersized = direct_search(r0, num_parts, h, dp, sim_dim, x_min, x_max, y_min, y_max, z_min,
                                                 z_max, pbc_option, pbc_x, pbc_y, pbc_z, ipn_pairs0, trial_size)
                trial_size += int(num_parts * 10)
            # t.toc('Direct search Cython')

        # Assign (0,0) to the first pair, and get rid of excess memory allocated to pairs.
        ipn_pairs0[0, 0] = ipn_pairs0[0, 1] = 0
        ipn_pairs0 = ipn_pairs0[np.where(ipn_pairs0[:, 0] >= 0)[0]]

        # Correct the relative positions if using periodic boundary conditions
        r_ij0 = r[ipn_pairs0[:, 0]] - r[ipn_pairs0[:, 1]]
        if pbc_option == 1:
            bc_object.periodic_bc(r0, r_ij0, x_min, x_max, y_min, y_max, z_min, z_max, dp, h, pbc_x, pbc_y, pbc_z,
                                  ipn_pairs0, 0)
        rij0 = np.linalg.norm(r_ij0, ord=None, axis=1, keepdims=True)

        # Calculate kernels.
        w0 = np.zeros((ipn0 + 1, 2))
        dw0 = np.zeros((ipn0 + 1, 2, 3))
        kernel_calcs(r_ij0, rij0, h, sim_dim, w0, dw0, kernel_type, ipn0)

        # ==============================================================================================================
        # ================================================= FILTERS ====================================================

        if c_n_option == 1:
            # t.tic()
            m = np.zeros((num_parts + 1, 3, 3), dtype=np.float64)
            kernel_deriv_moments(part_type, mass, rho0, r_ij0, ipn0, ipn_pairs0, dw0, m)
            kernel_grad_correction(ipn0, ipn_pairs0, dw0, m, sim_dim, num_parts, tol)
            # t.toc('Kernel Derivative Correction Cython')

        else:
            m = 1

        if n_order == 1:
            # t.tic()
            w_sum = np.zeros_like(part_type, dtype=np.float64)
            kernel_sum(part_type, mass, rho0, ipn_pairs0, ipn0, w0, w_sum)
            shepard_filter(ipn0, ipn_pairs0, w0, w_sum)
            # t.toc('Kernel renormalization Cython')

        else:
            w_sum = 1

        # ==============================================================================================================
        # Assigning the reference values to the particles' interaction parameters (for total Lagrangian approach).
        ipn = ipn0
        ipn_pairs = ipn_pairs0
        w = w0
        dw = dw0

    # ==================================================================================================================
    # ================================================ MAIN LOOP =======================================================
    # ==================================================================================================================

        while elapsed <= sim_time:

            print('Step', step)

            t.tic()

            # ==========================================================================================================
            # ================================ NNPS ALGORITHM AND KERNEL CALCULATION ===================================

            # This is used to control if the simulation will be a total Lagrangian or updated Lagrangian. For the total
            #  Lagrangian, we use a Lagrangian kernel (and material description of motion) and set update = 1e32. For an
            #  updated Lagrangian, we use an Eulerian kernel (and a spatial description of motion) and set update = n,
            #  where n is any integer greater than or equal to 1. If set equals to 1, update parameters every step.

            update = 1       # Updated Lagrangian.
            # update = 0        # Total Lagrangian.

            if update > 0 and step % update == 0 and step != 0:

                ipn = trial_size = int(min(sim_dim * 10 ** 6, num_parts * (4 * h / dp) ** sim_dim, ipn + num_parts))
                ipn_pairs = -np.ones((trial_size, 2), dtype=int)
                undersized = 1

                # Linked-list nearest neighbor search.
                if nnps.upper() == 'LL':
                    cells_parts = np.zeros((num_cells + 1, num_parts + 1), dtype=int)
                    parts_cells = np.zeros(num_parts + 1, dtype=int)

                    # Map the particles to their respective cells and vice-versa.
                    max_index = allocate_particles(r, x_div, y_div, num_cells, h, num_parts, cells_parts, parts_cells,
                                                   dp)
                    cells_parts = np.ascontiguousarray(cells_parts[:, 1:max_index])  # Gets rid of the first column.

                    # t.tic()
                    while undersized == 1:
                        ipn_pairs = -np.ones((trial_size, 2), dtype=int)
                        ipn, undersized = linked_list(r, parts_cells, cells_parts, grid, h, dp, sim_dim, num_parts,
                                                      ipn_pairs, trial_size, x_min, x_max, y_min, y_max, z_min, z_max,
                                                      pbc_option, pbc_x, pbc_y, pbc_z)
                        trial_size += int(num_parts * 10)
                    # t.toc('Linked-List Algorithm Cython')

                # Direct nearest neighbor search.
                else:
                    # t.tic()
                    while undersized == 1:
                        ipn_pairs = -np.ones((trial_size, 2), dtype=int)
                        ipn, undersized = direct_search(r, num_parts, h, dp, sim_dim, x_min, x_max, y_min, y_max, z_min,
                                                        z_max, pbc_option, pbc_x, pbc_y, pbc_z, ipn_pairs, trial_size)
                        trial_size += int(num_parts * 10)
                    # t.toc('Direct search Cython')

                ipn_pairs[0, 0] = ipn_pairs[0, 1] = 0
                ipn_pairs = ipn_pairs[np.where(ipn_pairs[:, 0] >= 0)[0]]

                r_ij = r[ipn_pairs[:, 0]] - r[ipn_pairs[:, 1]]
                if pbc_option == 1:
                    bc_object.periodic_bc(r, r_ij, x_min, x_max, y_min, y_max, z_min, z_max, dp, h, pbc_x, pbc_y, pbc_z,
                                          ipn_pairs, 0)
                rij = np.linalg.norm(r_ij, ord=None, axis=1, keepdims=True)

                w = np.zeros((ipn + 1, 2))
                dw = np.zeros((ipn + 1, 2, 3))
                kernel_calcs(r_ij, rij, h, sim_dim, w, dw, kernel_type, ipn)

                # ======================================================================================================
                # Updated reference properties to the ones at the beginning of the current time step.
                rho /= j                # Initial mass density.
                f_tot[:] = np.identity(3)  # Initialize relative deformation gradient.
                un = np.copy(u)         # Initialize displacement for the updated reference configuration.

                # ======================================================================================================
                # ============================================= FILTERS ================================================

                # Corrected-normalized kernel gradient as proposed in Bonet and Lok (1999).
                if c_n_option == 1:
                    # t.tic()
                    kernel_deriv_moments(part_type, mass, rho, r_ij, ipn, ipn_pairs, dw, m)
                    kernel_grad_correction(ipn, ipn_pairs, dw, m, sim_dim, num_parts, tol)
                    # t.toc('Kernel Derivative Correction Cython')

                if n_order == 1:
                    # t.tic()
                    kernel_sum(part_type, mass, rho, ipn_pairs, ipn, w, w_sum)
                    shepard_filter(ipn, ipn_pairs, w, w_sum)
                    # t.toc('Kernel renormalization Cython')

            # ==========================================================================================================
            # ==========================================================================================================

            # In the comments of the lines that follow it is to be understood that n refers to the current step, also
            #  the value of the variable at the end of the previous time step, or the beginning of the current one,
            #  while n+1 refers to the value of the variable at the end of the current step, or beginning of the next
            #  one.

            # ==========================================================================================================
            # ========================================= INITIAL PARTICLE UPDATE ========================================
            # This is necessary to update the velocity and position to the end of the first step, taking into account
            #  the initial state of stress in the material. This step is required despite each time integration scheme
            #  is used.

            # This signals the next module to calculate the acceleration and rate of change of energy based on
            #  the initial state of stress. Do not change the value of this parameter!!!
            flag = 0  # DO NOT CHANGE!!!

            if step == 0:

                flag = 1  # DO NOT CHANGE OR ERASE THIS!!!

                # Initial displacement vector (zero vector for every particle) and relative velocity.
                du_ij0 = u[ipn_pairs0[:, 0]] - u[ipn_pairs0[:, 1]]
                v_ij0 = v[ipn_pairs0[:, 0]] - v[ipn_pairs0[:, 1]]

                step_object.single_step(part_type, body, mass, r0, r_ij0, rij0, du_ij0, v0, v_ij0, dv, rho, f_tot, f,
                                        j_tot, j, b, be, je, eps, eps_e, eps_e_dev, xi, nxn, nexne, kirchhoff, tau, e,
                                        s_e, k_e, p_e, h, dp, num_parts, ipn, ipn_pairs, w, dw, sim_dim, bound_parts, c,
                                        alpha, beta, bulk, shear, a_psi, a_phi, sy, sy0, hard_mod, gen_mod, dyn_visc,
                                        exm, eq_time, elapsed, dummy, kernel, flag, step_size, y_criterion)

                flag = 0  # DO NOT CHANGE OR ERASE THIS!!!

            # ==========================================================================================================
            # ============================================= PREDICTOR-CORRECTOR ========================================

            # This implementation follows the one presented by Monaghan (2005).
            if integration.upper() == 'PC':

                # ======================================================================================================
                # ================================ UPDATE VELOCITIES AT HALF TIME STEP =================================

                # Increment of velocity.
                dvt = dv[fluid] * step_size

                # Velocity at half of the current time step, n (so, v_n+1/2).
                v_half[fluid] = v[fluid] + dvt / 2

                # Velocity estimate at end of step.
                v[fluid] += dvt
                v_ij = v[ipn_pairs[:, 0]] - v[ipn_pairs[:, 1]]

                # ======================================================================================================
                # ========================= RELATIVE DISPLACEMENT WITHIN THE CURRENT TIME STEP =========================

                # Total displacement of each particle within the current time step, n.
                du = (v_half + x_vel) * step_size  # du = r_n+1 - r_n
                u += du
                u[fluid] += du[fluid]
                u_ij = u[ipn_pairs[:, 0]] - u[ipn_pairs[:, 1]]

                # ======================================================================================================
                # ====================== UPDATE PARTICLES' POSITIONS AT THE END OF THE TIME STEP =======================

                # Move particles with a corrected velocity, (v + x_vel), which is closer to the local average.
                if xsph.upper() == 'Y':
                    vh_ij = v_half[ipn_pairs[:, 0]] - v_half[ipn_pairs[:, 1]]
                    xsph_object.calcVelMod(part_type, x_vel, mass, rho, vh_ij, w, epsilon, ipn_pairs, num_parts)

                # Position at the end of the current time step, n (so r_n+1).
                r[fluid] += du[fluid]
                r_ij = r[ipn_pairs[:, 0]] - r[ipn_pairs[:, 1]]

                # ======================================================================================================
                # ===================================== APPLY BOUNDARY CONDITIONS ======================================

                # Correct the relative positions if using periodic boundary conditions
                if pbc_option == 1 and step != 0 and update != 1:
                    bc_object.periodic_bc(r, r_ij, x_min, x_max, y_min, y_max, z_min, z_max, dp, h, pbc_x, pbc_y, pbc_z,
                                          ipn_pairs, 1)

                rij = np.linalg.norm(r_ij, ord=None, axis=1, keepdims=True)

                # ======================================================================================================
                # ============================ CALCULATE DEFORMATIONS, STRESSES AND RATES ==============================

                # Call single step calculation (see notes).
                # t.tic()
                step_object.single_step(part_type, body, mass, r, r_ij, rij, u_ij, v, v_ij, dv, rho, f_tot, f, j_tot, j,
                                        b, be, je, eps, eps_e, eps_e_dev, xi, nxn, nexne, kirchhoff, tau, e, s_e, k_e,
                                        p_e, h, dp, num_parts, ipn, ipn_pairs, w, dw, sim_dim, bound_parts, c, alpha,
                                        beta, bulk, shear, a_psi, a_phi, sy, sy0, hard_mod, gen_mod, dyn_visc, exm,
                                        eq_time, elapsed, dummy, kernel, flag, step_size, y_criterion)
                # t.toc('Single step')
                # print()
                # ======================================================================================================
                # ========================= UPDATE VELOCITIES TO THE END OF THE TIME STEP ==============================

                # Velocity at half of the current time step, n (so, v_n+1).
                v[fluid] = v_half[fluid] + dv[fluid] * step_size / 2
                v[np.abs(v) < 1e-6] = 0

            # ==========================================================================================================
            # ============================================ LEAP-FROG INTEGRATION =======================================

            elif integration.upper() == 'LF':

                # ======================================================================================================
                # ================================ UPDATE VELOCITIES AT HALF TIME STEP =================================

                # Increment of velocity.
                dvt = dv[fluid] * step_size

                # Velocity at half of the current time step, n (so, v_n+1/2).
                v_half[fluid] += dvt
                if step == 0:
                    v_half[fluid] -= dvt / 2

                # Velocity estimate at end of step.
                v[fluid] += dvt
                v_ij = v[ipn_pairs[:, 0]] - v[ipn_pairs[:, 1]]

                # ======================================================================================================
                # ========================= RELATIVE DISPLACEMENT WITHIN THE CURRENT TIME STEP =========================

                # Total displacement of each particle within the current time step, n.
                du = (v_half + x_vel) * step_size  # du = r_n+1 - r_n
                u += du
                u[fluid] += du[fluid]
                u_ij = u[ipn_pairs[:, 0]] - u[ipn_pairs[:, 1]]

                # ======================================================================================================
                # ====================== UPDATE PARTICLES' POSITIONS AT THE END OF THE TIME STEP =======================

                # Move particles with a corrected velocity, (v + x_vel), which is closer to the local average.
                if xsph.upper() == 'Y':
                    vh_ij = v_half[ipn_pairs[:, 0]] - v_half[ipn_pairs[:, 1]]
                    xsph_object.calcVelMod(part_type, x_vel, mass, rho, vh_ij, w, epsilon, ipn_pairs, num_parts)

                # Position at the end of the current time step, n (so r_n+1).
                r[fluid] += du[fluid]
                r_ij = r[ipn_pairs[:, 0]] - r[ipn_pairs[:, 1]]

                # ======================================================================================================
                # ===================================== APPLY BOUNDARY CONDITIONS ======================================

                # Correct the relative positions if using periodic boundary conditions
                if pbc_option == 1 and step != 0 and update != 1:
                    bc_object.periodic_bc(r, r_ij, x_min, x_max, y_min, y_max, z_min, z_max, dp, h, pbc_x, pbc_y, pbc_z,
                                          ipn_pairs, 1)

                rij = np.linalg.norm(r_ij, ord=None, axis=1, keepdims=True)

                # ======================================================================================================
                # ============================ CALCULATE DEFORMATIONS, STRESSES AND RATES ==============================

                # Call single step calculation (see notes).
                # t.tic()
                step_object.single_step(part_type, body, mass, r, r_ij, rij, u_ij, v, v_ij, dv, rho, f_tot, f, j_tot, j,
                                        b, be, je, eps, eps_e, eps_e_dev, xi, nxn, nexne, kirchhoff, tau, e, s_e, k_e,
                                        p_e, h, dp, num_parts, ipn, ipn_pairs, w, dw, sim_dim, bound_parts, c, alpha,
                                        beta, bulk, shear, a_psi, a_phi, sy, sy0, hard_mod, gen_mod, dyn_visc, exm,
                                        eq_time, elapsed, dummy, kernel, flag, step_size, y_criterion)
                # t.toc('Single step')
                # print()
                # ======================================================================================================
                # ========================= UPDATE VELOCITIES TO THE END OF THE TIME STEP ==============================

                # Velocity at half of the current time step, n (so, v_n+1).
                v[fluid] = v_half[fluid] + dv[fluid] * step_size / 2
                v[np.abs(v) < 1e-6] = 0

            # ==========================================================================================================
            # ===================================== EXPLICIT FORWARD EULER INTEGRATION =================================

            else:

                # t.tic()
                # ======================================================================================================
                # ========================= UPDATE VELOCITIES TO THE END OF THE TIME STEP ==============================

                # Velocity at the end of the current time step, n (so, v_n+1).
                v[fluid] += dv[fluid] * step_size
                v_ij = v[ipn_pairs[:, 0]] - v[ipn_pairs[:, 1]]

                # ======================================================================================================
                # ====================== UPDATE PARTICLES' POSITIONS AT THE END OF THE TIME STEP =======================

                # Move particles with a corrected velocity, (v + x_vel), which is closer to the local average.
                if xsph.upper() == 'Y':
                    xsph_object.calcVelMod(part_type, x_vel, mass, rho, v_ij, w, epsilon, ipn_pairs, num_parts)

                du = (v + x_vel) * step_size  # du = r_n+1 - r_n
                r += du

                # ======================================================================================================
                # ===================== CORRECT POSITION OF PARTICLES IF THEY LEAVE THE DOMAIN =========================

                # t.tic()
                # Takes particles that went out of the problem domain and gets rid of them.
                if nnps.upper() == 'LL' and pbc_option == 0:

                    # Particles whose x coordinate exceeded the maximum x coordinate of the domain.
                    xl = np.where(r[:, 0] < dx_min)[0]
                    xr = np.where(r[:, 0] > dx_max)[0]
                    xout = np.union1d(xl, xr)

                    # Particles whose y coordinate exceeded the maximum y coordinate of the domain.
                    yu = np.where(r[:, 1] > dy_max)[0]
                    yb = np.where(r[:, 1] < dy_min)[0]
                    yout = np.union1d(yu, yb)

                    # Particles that went outside of the domain.
                    out = np.union1d(xout, yout)

                    part_type[out] = 4
                    r[out, :] = 0
                    v[out, :] = 0

                # t.toc('Boundary checking')

                # ======================================================================================================
                # ===================================== APPLY BOUNDARY CONDITIONS ======================================

                r_ij = r[ipn_pairs[:, 0]] - r[ipn_pairs[:, 1]]

                # Correct the relative positions if using periodic boundary conditions
                if pbc_option == 1:
                    bc_object.periodic_bc(r, r_ij, x_min, x_max, y_min, y_max, z_min, z_max, dp, h, pbc_x, pbc_y, pbc_z,
                                          ipn_pairs, 1)

                rij = np.linalg.norm(r_ij, ord=None, axis=1, keepdims=True)

                # ======================================================================================================
                # ========================= RELATIVE DISPLACEMENT WITHIN THE CURRENT TIME STEP =========================

                # Total displacement of each particle within the current time step, n.
                u += du

                # Correct relative displacement between particles for total or partially updated Lagrangian approaches.
                if update != 1:
                    du = u - un

                du_ij = du[ipn_pairs[:, 0]] - du[ipn_pairs[:, 1]]
                du_ij[np.abs(du_ij) < tol] = 0
                du_ij[:] = np.round(du_ij[:], decimals=digits)
                # t.toc('Updating r, v, and u')

                # ======================================================================================================
                # ============================ CALCULATE DEFORMATIONS, STRESSES AND RATES ==============================

                # Call single step calculation (see notes).
                # t.tic()
                step_object.single_step(part_type, body, mass, r, r_ij, rij, du_ij, v, v_ij, dv, rho, f_tot, f, j_tot,
                                        j, b, be, je, eps, eps_e, eps_e_dev, xi, nxn, nexne, kirchhoff, tau, e, s_e,
                                        k_e, p_e, h, dp, num_parts, ipn, ipn_pairs, w, dw, sim_dim, bound_parts, c,
                                        alpha, beta, bulk, shear, a_psi, a_phi, sy, sy0, hard_mod, gen_mod, dyn_visc,
                                        exm, eq_time, elapsed, dummy, kernel, flag, step_size, y_criterion)
                # t.toc('Single step')
                # print()

        # ==============================================================================================================
        # =============================================== OUTPUT RESULTS ===============================================
        # ==============================================================================================================

            # Output all information of the current configuration every m steps.
            # t.tic()
            freq = 1
            if step % freq == 0:
                file_num += 1
                # Update mass density.
                newrho = np.copy(rho) / j
                output_object.create_csv_file(r, u, v, newrho, s_e, k_e, kirchhoff, eps, eps_e, num_parts, nxn, nexne,
                                              file_num, version)
            # t.toc()
            print('Simulation ', round((elapsed / sim_time) * 100, 2), "% Complete!")
            print('Elapsed time: ', round(elapsed, 6), "s")

            elapsed += step_size
            step += 1
            t.toc('Step')
            print()

            # if step == 5:
            #     exit()

# ======================================================================================================================
