from __future__ import print_function
__author__ = 'alomir'

import numpy as np

# This class implements the strain and stress update for a given initial trial strain and stress state. It can handle
#  nonlinear hardening and viscoplastic constitutive models using Pierce viscoplastic consistency equation. The
#  nonlinear solver is a classic Newton-Raphson scheme with a single equation. For details of the implementation, see
#  chapters 8 and 11 of the book of Souze Neto et al. (2008)


class ViscoPlasticity:

    def __init__(self, elmodel='HKY', ycriterion='VM', cmodel='EP'):
        self.yieldFunc = ycriterion
        self.constModel = cmodel
        self.elasticity = elmodel

    # ==================================================================================================================
    # This method implements the yield function calculation for models based on J2 plasticity, which can be written in
    #  the following format: f = sqrt(3*J2) + aphi*p - Sig_y.
    @staticmethod
    def yield_criterion(j2, p, aphi, sig_y):
        return np.sqrt(3 * j2) + aphi * p - sig_y

    # ==================================================================================================================
    # This method implements the hardening modulus and yield stress parameter update based on an assumed internal
    #  energy dissipation potential of cubic order: w = l1*xi^3 + l2*xi^2 + l3*xi + l4.
    @staticmethod
    def hardening(xi, h_prime, sig_y0, l1=0.0):

        l2 = h_prime / 2
        l3 = sig_y0

        # Yield stress.
        sig_y = 3 * l1 * np.power(xi, 2) + 2 * l2 * xi + l3

        # Generalized hardening modulus.
        hard_mod = 6 * l1 * xi + 2 * l2

        return sig_y, hard_mod

    # ==================================================================================================================
    # This method implements the standard Newton-Raphson nonlinear solver for the update of the principal elastic
    #  stretches and the Kirchhoff stress tensor for rate-dependent plasticity (viscoplasticity). If eta or m are
    #  zero, it recovers the rate-independent response, and if eta is very large, in the limit, recovers the purely
    #  elastic response.
    def solver(self, eps, eps_dev, je, xi, tau, bulk, shear, c_e, aphi, apsi, s_y, s_y0, hard, h_prime, eta, m, s_e,
               num_parts, stress_object, elast_object, flag, step_size, tol, digits):

        # Tolerance for "zero".
        err = 1e-12
        kmax = 15
        np.set_printoptions(precision=16)

        # Default values for elastoplastic model and Von Mises yield criterion.
        if self.constModel == 'EP':
            eta = m = 0
        if self.yieldFunc == 'VM':
            aphi = apsi = 0

        # Deviatoric part of trial Tau.
        tau_dev = np.copy(tau)
        p = np.zeros((num_parts + 1, 1))
        stress_object.stress_tensor_decomposition(tau, tau_dev, p, np.arange(num_parts + 1))

        # The second invariant and the norm of trial Tau.
        j2 = np.zeros((num_parts + 1, 1))
        stress_object.calc_j2(tau_dev, j2, np.arange(num_parts + 1))

        # Verify if elastic step.
        y = self.yield_criterion(j2, p, aphi, s_y)

        # Check if there are particles that violated the yield criterion.
        plast_parts = np.where(y >= err)[0]
        size = np.size(plast_parts)

        if size > 0:

            # Initializing variables.
            xi_k = np.copy(xi)
            dg = np.zeros((num_parts + 1, 1))

            # Initial range of particles: updated to eliminate particles that have converged already.
            parts = np.copy(plast_parts)

            # ======================================== RETURN TO THE SMOOTH CONE =======================================

            k = 0
            while size > 0 and k <= kmax:

                # Equivalent stress.
                sig_e = np.sqrt(3 * j2[parts]) - 3 * shear * dg[parts] + aphi * p[parts] - aphi * apsi * bulk * \
                        dg[parts]

                # Auxiliary viscosity parameters.
                a1 = 1 / (dg[parts] * eta + step_size)
                a2 = np.power(step_size * a1, m)

                # Residual.
                r = sig_e * a2 - s_y[parts]

                # Residual derivative.
                dr = -((3 * shear + aphi * apsi * bulk + sig_e * eta * m * a1) * a2 + hard[parts])

                # Consistency parameter increment and update.
                ddg = -r / dr
                dg[parts] += ddg

                # Update plastic internal variable.
                xi_k[parts] = xi[parts] + dg[parts]

                # Update hardening parameters, yield stress and hardening modulus.
                s_y[parts], hard[parts] = self.hardening(xi_k[parts], h_prime, s_y0[parts])

                # Update particles that still did not converge and iteration counter.
                fail = np.where(np.abs(ddg) > err)[0]
                parts = parts[fail]
                size = np.size(parts)
                k += 1

            # Update the deviatoric components of the principal stretches.
            eps_dev[plast_parts] *= (1 - np.sqrt(3) * shear * dg[plast_parts] / np.sqrt(j2[plast_parts]))
            eps_dev[np.abs(eps_dev) < tol] = 0

            # Volumetric components of strain.
            tr_eps_tr = np.sum(eps[plast_parts], 1, keepdims=True)
            eps_vol = (tr_eps_tr - dg[plast_parts] * apsi) / 3

            # ============================================ RETURN TO THE APEX ==========================================

            if self.yieldFunc == 'DP':

                # Parts that returned to the imaginary part of the cone.
                apex = np.where(np.sqrt(3 * j2[plast_parts]) - 3 * dg[plast_parts] * shear < err)[0]
                parts = plast_parts[apex]  # Used to zero the deviatoric stretches.
                size = np.size(parts)

                if size > 0:

                    # Set to zero the deviatoric components of the particles whose stress are on the apex.
                    eps_dev[parts] = 0

                    if apsi > 0:

                        # Reinitialize the consistency parameter and the iteration counter.
                        dg[parts] = 0
                        k = 0

                        print('Called apex return')
                        print()

                        while size > 0 and k <= kmax:
                            # Equivalent stress.
                            sig_e = aphi * p[parts] - aphi * apsi * bulk * dg[parts]

                            # Auxiliary viscosity parameters.
                            a1 = 1 / (dg[parts] * eta + step_size)
                            a2 = np.power(step_size * a1, m)

                            # Residual.
                            r = sig_e * a2 - s_y[parts]

                            # Residual derivative.
                            dr = -((aphi * apsi * bulk + sig_e * eta * m * a1) * a2 + hard[parts])

                            # Consistency parameter increment and update.
                            ddg = -r / dr
                            dg[parts] += ddg

                            # Update plastic internal variable.
                            xi_k[parts] = xi[parts] + dg[parts]

                            # Update hardening parameters, yield stress and hardening modulus.
                            s_y[parts], hard[parts] = self.hardening(xi_k[parts], h_prime, s_y0[parts])

                            # Update particles that still did not converge and iteration counter.
                            apex = np.where(np.abs(ddg) > err)[0]
                            parts = parts[apex]
                            size = np.size(parts)
                            k += 1

                        eps_vol = (tr_eps_tr - dg[plast_parts] * apsi) / 3

                    # In this case, if psi = 0 and the apex return is called, there is no physical state of deformation
                    #  feasible (not supported by the yield criterion). What we do here then, is to "freeze" the elastic
                    #  strain and set the plastic strains to zero. This mimics a behavior such that the particle is
                    #  isolated from the neighbor particles if this state is reached.
                    else:
                        dg[plast_parts[apex]] = 0
                        eps_vol[apex, :] = s_y[plast_parts[apex]] / (3 * aphi * bulk)

            # ============================================ FINAL CALCULATIONS ==========================================

            # Update principal logarithmic stretches.
            eps[plast_parts] = eps_dev[plast_parts] + eps_vol
            eps[np.abs(eps) < tol] = 0
            tr_eps = np.sum(eps[plast_parts], 1, keepdims=True)
            je[plast_parts] = np.exp(tr_eps)

            # Update stress.
            elast_object.kirchhoff_stress(eps, eps_dev, je, tau, s_e, bulk, shear, c_e, self.elasticity, plast_parts)
            tau[np.abs(tau) < tol] = 0

            # Update plastic strain-like internal variables.
            xi[plast_parts] = xi_k[plast_parts]
            xi[xi < tol] = 0

            # ====================================== ROUND UP OPERATIONS ===============================================

            eps[:] = np.round(eps[:], decimals=digits)
            eps_dev[:] = np.round(eps_dev[:], decimals=digits)
            xi[:] = np.round(xi[:], decimals=digits)
            tau[:] = np.round(tau[:], decimals=digits)

            # ==========================================================================================================

            # Flag to tell the code to reconstruct the strain tensors based on the updated principal stretches.
            flag[plast_parts] = 1

# ======================================================================================================================
    # This method implements the closed form solver for the update of the principal elastic stretches and the Kirchhoff
    #  stress tensor for rate-independent plasticity.
    def solver_exact(self, eps, eps_dev, je, xi, tau, bulk, shear, c_e, aphi, apsi, s_y, s_y0, hard, h_prime, s_e,
                     num_parts, stress_object, elast_object, flag, tol, digits):

        # Tolerance for "zero".
        err = 1e-12
        np.set_printoptions(precision=16)

        # Default values for elastoplastic model and Von Mises yield criterion.
        if self.yieldFunc == 'VM':
            aphi = apsi = 0

        # Deviatoric part of trial Tau.
        tau_dev = np.copy(tau)
        p = np.zeros((num_parts + 1, 1))
        stress_object.stress_tensor_decomposition(tau, tau_dev, p, np.arange(num_parts + 1))

        # The second invariant and the norm of trial Tau.
        j2 = np.zeros((num_parts + 1, 1))
        stress_object.calc_j2(tau_dev, j2, np.arange(num_parts + 1))

        # Verify if elastic step.
        y = self.yield_criterion(j2, p, aphi, s_y)

        # Check if there are particles that violated the yield criterion.
        parts = np.where(y >= err)[0]
        size = np.size(parts)

        if size > 0:

            # ======================================== RETURN TO THE SMOOTH CONE =======================================

            # Consistency parameter increment and update.
            dg = y[parts] / (3 * shear + aphi * apsi * bulk + hard[parts])

            # Update plastic internal variable.
            xi[parts] += dg

            # Update hardening parameters, yield stress and hardening modulus.
            s_y[parts], hard[parts] = self.hardening(xi[parts], h_prime, s_y0[parts])

            # Update the deviatoric components of the principal stretches.
            eps_dev[parts] *= (1 - np.sqrt(3) * shear * dg / np.sqrt(j2[parts]))
            eps_dev[np.abs(eps_dev) < tol] = 0

            # Volumetric components of strain.
            tr_eps_tr = np.sum(eps[parts], 1, keepdims=True)
            eps_vol = (tr_eps_tr - dg * apsi) / 3

            # ============================================ RETURN TO THE APEX ==========================================

            apex = np.array([])
            if self.yieldFunc == 'DP':

                # Parts that returned to the imaginary part of the cone.
                apex = np.where(np.sqrt(3 * j2[parts]) - 3 * dg * shear < err)[0]
                parts = parts[apex]  # Used to zero the deviatoric stretches.
                size = np.size(parts)

                if size > 0:

                    # Set to zero the deviatoric components of the particles whose stress are on the apex.
                    eps_dev[parts, :] = 0

                    if apsi > 0:
                        print('Called apex return')
                        print()

                        # Consistency parameter increment and update.
                        dg = y[parts] / (aphi * apsi * bulk + hard[parts])

                        # Update plastic internal variable.
                        xi[parts] += dg

                        # Update hardening parameters, yield stress and hardening modulus.
                        s_y[parts], hard[parts] = self.hardening(xi[parts], h_prime, s_y0[parts])

                        # Update volumetric stretch.
                        eps_vol[apex] = (tr_eps_tr[apex] - dg * apsi) / 3

            # ============================================ FINAL CALCULATIONS ==========================================

            # Update principal logarithmic stretches.
            eps[parts] = eps_dev[parts] + eps_vol[apex]
            eps[np.abs(eps) < tol] = 0
            tr_eps = np.sum(eps[parts], 1, keepdims=True)
            je[parts] = np.exp(tr_eps)

            # Update stress.
            elast_object.kirchhoff_stress(eps, eps_dev, je, tau, s_e, bulk, shear, c_e, self.elasticity, parts)
            tau[np.abs(tau) < tol] = 0

            # Update plastic strain-like internal variables.
            xi[xi < tol] = 0

            # ====================================== ROUND UP OPERATIONS ===============================================

            eps[:] = np.round(eps[:], decimals=digits)
            eps_dev[:] = np.round(eps_dev[:], decimals=digits)
            xi[:] = np.round(xi[:], decimals=digits)
            tau[:] = np.round(tau[:], decimals=digits)

            # ==========================================================================================================

            # Flag to tell the code to reconstruct the strain tensors based on the updated principal stretches.
            flag[parts] = 1

# ======================================================================================================================
