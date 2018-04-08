from __future__ import print_function
__author__ = 'alomir'

import math
import numpy as np


class Initialization:

    def __init__(self, sim_params, init_r):
        self.user_params = sim_params
        self.init_pos = init_r

    # Returns initial interparticle distance. Assumes it is the greatest initial distance or that all particle are
    #  equidistant.
    def get_init_part_dist(self):
        return float(self.user_params['dp'])

    # Returns the value of the initial ratio of the smooth length to initial particle distance. Assumed constant for all
    #  particles.
    def get_smooth_factor(self):
        return float(self.user_params['kh'])

    # Return the initial smoothLength for all particles. Assumed homogeneous.
    def get_h(self):
        return self.get_init_part_dist() * self.get_smooth_factor()

    # Returns the simulation time.
    def get_sim_time(self):
        return float(self.user_params['simTime'])

    # Returns the initial step size. Assume equal for all particles.
    def get_dt0(self):
        return float(self.user_params['stepSize'])

    # Returns the number or steps based on the simulation time and initial step size.
    def get_num_steps(self):
        return int(self.get_sim_time() / self.get_dt0())

    # Return the kernel choice. W for Wendland C6, S for cubic spline, and QS for quintic spline.
    def kernel_option(self):
        return str(self.user_params['kernel'])

    # Returns the number of dimensions of the simulation.
    def get_sim_dim(self):
        return int(self.user_params['simDim'])

    # Returns whether the first order correction will be applied to the kernel derivative.
    def kernel_correction_option(self):
        return int(self.user_params['CorrNorm'])

    # Returns whether renormalization of the kernel will be applied.
    def kernel_correction_order(self):
        return int(self.user_params['NOrder'])

    # Returns the choice for the nearest neighbor search algorithm to be used. LL for linked-list, and DS for direct
    # search.
    def nnps_option(self):
        return str(self.user_params['NNPS'])

    # Returns Monaghan's artificial viscosity parameter alpha.
    def get_artvisc_alpha(self):
        return float(self.user_params['alpha'])

        # Returns Monaghan's artificial viscosity parameter beta.
    def get_artvisc_beta(self):
        return float(self.user_params['beta'])

    # Return the artificial sound speed velocity used for calculations of the artificial viscosity and time stepping.
    def get_sound_speed(self):
        return float(self.user_params['c'])

    # Return the reference density of the particles, used in the EOS.
    def get_rho0(self):
        return float(self.user_params['rho0'])

    # Return the Cartesian coordinates of the problem's domain's vertices. Assumes it a "brick-shaped" domain, and that
    #  it is contained in the positive quadrant of the Cartesian axis.
    def get_domain(self):
        dp = self.get_init_part_dist()

        x_min, x_max, y_min, y_max, z_min, z_max = self.get_pbc_domain()

        dxmin = x_min - dp / 2
        dxmax = x_max + dp / 2
        dymin = y_min - dp / 2
        dymax = y_max + dp / 2
        dzmin = z_min - dp / 2
        dzmax = z_max + dp / 2

        if dxmin < 0:
            dxmin = 0
        if dymin < 0:
            dymin = 0
        if dzmin < 0:
            dzmin = 0

        return dxmin, dxmax, dymin, dymax, dzmin, dzmax

    # Returns the number of divisions of the domain in the x-, y- and z- directions used to create a grid for the
    #  linked-list search algorithm.
    def get_grid_divisions(self):
        nnps = self.nnps_option()
        pbc = self.get_pbc_option()
        pbc_x, pbc_y, pbc_z = self.get_pbc_directions()
        dx_min, dx_max, dy_min, dy_max, dz_min, dz_max = self.get_domain()
        kr = 2 * self.get_h()

        x_div = 1 + int((dx_max - dx_min) / kr)
        y_div = 1 + int((dy_max - dy_min) / kr)
        z_div = 1 + int((dz_max - dz_min) / kr)

        # Check to avoid an additional layer/row/column of empty cells.
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_pbc_domain()

        lx = x_div * kr - x_max
        ly = y_div * kr - y_max
        lz = z_div * kr - z_max

        if lx >= kr and x_div > 1:
            x_div -= 1
        if ly >= kr and y_div > 1:
            y_div -= 1
        if lz >= kr and z_div > 1:
            z_div -= 1

        if nnps.upper() == 'LL' and pbc == 1:
            if pbc_x == 1 and x_div < 4:
                print()
                print('################################################################################################'
                      '################################################')
                print('Insufficient number of particles in the x-direction for periodic boundary conditions. Please, '
                      'increase domain size by at least 2h and try again.')
                print('################################################################################################'
                      '################################################')
                exit()
            if pbc_y == 1 and y_div < 4:
                print()
                print('################################################################################################'
                      '################################################')
                print('Insufficient number of particles in the y-direction for periodic boundary conditions. Please, '
                      'increase domain size by at least 2h and try again.')
                print('################################################################################################'
                      '################################################')
                exit()
            if pbc_z == 1 and z_div < 4:
                print()
                print('################################################################################################'
                      '################################################')
                print('Insufficient number of particles in the z-direction for periodic boundary conditions. Please, '
                      'increase domain size by at least 2h and try again.')
                print('################################################################################################'
                      '################################################')
                exit()

        return int(x_div), int(y_div), int(z_div)

    # Return the number of cells in the grid formed for the Linked-List search algorithm. Assumes all particles have
    #  the same smooth length.
    def get_num_cells(self):
        x_div, y_div, z_div = self.get_grid_divisions()

        return x_div * y_div * z_div

    # Return the choice for whether using the XSPH approach or not.
    def xsph_option(self):
        return str(self.user_params['XSPH'])

    # Return the value of epsilon used in the XSPH approach.
    def get_xsph_epsilon(self):
        return float(self.user_params['epsilon'])

    # Return the integration scheme to be used. PC - Predictor-Corrector, LF - Leap-Frog, E - Forward Euler.
    def integration_scheme_option(self):
        return str(self.user_params['integration'])

    # Return the Mohr-Coulomb parameters, cohesion in Pascal and friction angle in radians.
    def get_mohr_coulomb_params(self):
        return float(self.user_params['cohesion']), math.radians(float(self.user_params['phi']))

    # Return the dynamic viscosity of the fluid under consideration in the problem.
    def get_dynamic_visc(self):
        return float(self.user_params['mu0'])

    # Returns if periodic boundary conditions are being used.
    def get_pbc_option(self):
        return int(self.user_params['PBC'])

    # Return if dummy boundary particles are being used.
    def dummy_particles_option(self):
        return int(self.user_params['dummy'])

    # Return if the condition is slip or no slip for the solid boundary.
    def slip_condition_option(self):
        if self.dummy_particles_option() == 1:
            return str(self.user_params['noslip'])
        else:
            return 'N'

    # Return which coordinate directions are periodic.
    def get_pbc_directions(self):
        if self.get_pbc_option() == 1:

            pbc_x = int(self.user_params['PBCX'])
            pbc_y = int(self.user_params['PBCY'])
            pbc_z = int(self.user_params['PBCZ'])

            return pbc_x, pbc_y, pbc_z

        return int(0), int(0), int(0)

    # Return the initial minimum and maximum coordinates of the problem..
    def get_pbc_domain(self):
        pbc_xmin = float(np.min(self.init_pos[1:, 0]))
        pbc_xmax = float(np.max(self.init_pos[1:, 0]))
        pbc_ymin = float(np.min(self.init_pos[1:, 1]))
        pbc_ymax = float(np.max(self.init_pos[1:, 1]))
        pbc_zmin = float(np.min(self.init_pos[1:, 2]))
        pbc_zmax = float(np.max(self.init_pos[1:, 2]))

        return pbc_xmin, pbc_xmax, pbc_ymin, pbc_ymax, pbc_zmin, pbc_zmax

    # Return if the problem is an initial hydrostatic equilibration_option problem.
    def equilibration_option(self):
        return str(self.user_params['Equilibration'])

    # Return the equilibration_option time.
    def get_equilibration_time(self):
        if self.equilibration_option().upper() == 'Y':
            return float(self.user_params['eqTime'])
        else:
            return 0.0

    # Return the Druker-Prager yield criterion parameters.
    def get_drucker_prager_params(self):

        # Simulation dimensions. If 2, assumed plane strain.
        sim_dim = int(self.get_sim_dim())

        # Cohesion and internal friction angle of the material.
        cohesion, phi = self.get_mohr_coulomb_params()

        # Dilation angle of the material.
        psi = math.radians(float(self.user_params['Dilation_angle']))

        # Drucker-Prager dilation, friction and cohesion material parameter.
        if sim_dim <= 2:
            a_psi = 3 * math.sqrt(3) * math.tan(psi) / (math.sqrt(9 + 12 * math.pow(math.tan(psi), 2)))
            a_phi = 3 * math.sqrt(3) * math.tan(phi) / (math.sqrt(9 + 12 * math.pow(math.tan(phi), 2)))
            kc = 3 * math.sqrt(3) / (math.sqrt(9 + 12 * math.pow(math.tan(phi), 2)))
        else:
            a_psi = 6 * math.sin(psi) / (3 - math.sin(psi))
            a_phi = 6 * math.sin(phi) / (3 - math.sin(phi))
            kc = 6 * math.cos(phi) / (3 - math.sin(psi))

        return a_psi, a_phi, kc

    # Return the elastic parameters of the material.
    def get_elastic_params(self):
        young = float(self.user_params['Young_modulus'])
        nu = float(self.user_params['Poisson_ratio'])

        bulk = young / (3 * (1 - 2 * nu))
        shear = young / (2 * (1 + nu))

        return young, nu, bulk, shear
