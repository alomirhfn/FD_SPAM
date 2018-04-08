# This file implements the class for outputting the results of the simulation into files with extension CSV.

__author__ = 'alomir'

import csv
import numpy as np


class Output:

    # Constructor of the output class that receive the following parameters:
    #  folderPath = address of the folder where the out put should be saved to.
    #  currStep = the number of the step which data will be written to file.
    def __init__(self, folder_path):
        self.path = folder_path

# ======================================================================================================================
    # Method that writes other information to file. It can be modified accordingly.
    def create_csv_file(self, r, u, v, rho, s_e, k_e, sigma, eps, eps_e, num_parts, nxn, nexne, n, version):

        # Total displacement of each particle by the end of the current step.
        norm_u = np.linalg.norm(u, ord=None, axis=1)

        # Norm of the velocity.
        norm_v = np.linalg.norm(v, ord=None, axis=1)

        if n == 0:
            e_tot = nxn[:, 0]
            e_el = nexne[:, 0]
        else:
            # Convert total logarithmic strain tensor from principal to the reference system of coordinates.
            e_tot = 2 * np.sum(np.reshape(eps, ((num_parts + 1), 3, 1, 1)) * nxn, 1)
            e_tot[:, 0, 0] *= 0.5
            e_tot[:, 1, 1] *= 0.5
            e_tot[:, 2, 2] *= 0.5

            # Convert logarithmic elastic strain tensor from principal to the reference system of coordinates.
            e_el = 2 * np.sum(np.reshape(eps_e, ((num_parts + 1), 3, 1, 1)) * nexne, 1)
            e_el[:, 0, 0] *= 0.5
            e_el[:, 1, 1] *= 0.5
            e_el[:, 2, 2] *= 0.5

        # Plastic logarithmic strain tensor.
        e_pl = e_tot - e_el

        # Deviatoric part of sigma
        tr_sigma = np.reshape(sigma[:, 0, 0] + sigma[:, 1, 1] + sigma[:, 2, 2], (num_parts + 1, 1, 1))
        p = tr_sigma / 3
        s = sigma[:] - p * np.reshape(np.tile(np.identity(3), (num_parts + 1, 1)), (num_parts + 1, 3, 3))

        # Von Mises equivalent stress, q.
        q = np.sqrt(3 * np.sum(np.sum(s * s, 2), 1) / 2)
        p = -p[:, 0, 0]   # Hydrostatic compressive pressure is positive.

        # Full version of output.
        if version == 'F':

            step_data = [['ID', 'x', 'y', 'z', 'ux', 'uy', 'uz', '|u|', 'vx', 'vy', 'vz', '|v|', 'rho', 'sw', 'kw',
                          'Sxx', 'Syy', 'Szz', 'Sxy', 'Sxz', 'Syz', 'Exx', 'Eyy', 'Ezz', 'Exy', 'Exz', 'Eyz', 'Eelxx',
                          'Eelyy', 'Eelzz', 'Eelxy', 'Eelxz', 'Eelyz', 'Eplxx', 'Eplyy', 'Eplzz', 'Eplxy', 'Eplxz',
                          'Eplyz', 'q', 'p']]

            for i in range(1, num_parts + 1):
                new_data_line = [i, r[i, 0], r[i, 1], r[i, 2], u[i, 0], u[i, 1], u[i, 2], norm_u[i], v[i, 0], v[i, 1],
                                 v[i, 2], norm_v[i], rho[i, 0], s_e[i, 0], k_e[i, 0], sigma[i, 0, 0], sigma[i, 1, 1],
                                 sigma[i, 2, 2], sigma[i, 0, 1], sigma[i, 0, 2], sigma[i, 1, 2], e_tot[i, 0, 0],
                                 e_tot[i, 1, 1], e_tot[i, 2, 2], e_tot[i, 0, 1], e_tot[i, 0, 2], e_tot[i, 1, 2],
                                 e_el[i, 0, 0], e_el[i, 1, 1], e_el[i, 2, 2], e_el[i, 0, 1], e_el[i, 0, 2],
                                 e_el[i, 1, 2], e_pl[i, 0, 0], e_pl[i, 1, 1], e_pl[i, 2, 2], e_pl[i, 0, 1],
                                 e_pl[i, 0, 2], e_pl[i, 1, 2], q[i], p[i]]

                step_data.append(new_data_line)

        # Simplified version of output (with version = 'S').
        else:

            step_data = [['x', 'y', 'z', 'rho', 'Sxx', 'Syy', 'Szz', 'Sxy', 'Sxz', 'Syz', 'Exx', 'Eyy', 'Ezz', 'Exy',
                          'Exz', 'Eyz', 'Eelxx', 'Eelyy', 'Eelzz', 'Eelxy', 'Eelxz', 'Eelyz', 'Eplxx', 'Eplyy', 'Eplzz',
                          'Eplxy', 'Eplxz', 'Eplyz', 'q', 'p']]

            for i in range(1, num_parts + 1):
                new_data_line = [r[i, 0], r[i, 1], r[i, 2], rho[i, 0], sigma[i, 0, 0], sigma[i, 1, 1], sigma[i, 2, 2],
                                 sigma[i, 0, 1], sigma[i, 0, 2], sigma[i, 1, 2], e_tot[i, 0, 0], e_tot[i, 1, 1],
                                 e_tot[i, 2, 2], e_tot[i, 0, 1], e_tot[i, 0, 2], e_tot[i, 1, 2], e_el[i, 0, 0],
                                 e_el[i, 1, 1], e_el[i, 2, 2], e_el[i, 0, 1], e_el[i, 0, 2], e_el[i, 1, 2],
                                 e_pl[i, 0, 0], e_pl[i, 1, 1], e_pl[i, 2, 2], e_pl[i, 0, 1], e_pl[i, 0, 2],
                                 e_pl[i, 1, 2], q[i], p[i]]

                step_data.append(new_data_line)

        print("Outputting Results...")
        with open(self.path + 'N' + str(n) + '.csv', 'wb') as fp:
                writer = csv.writer(fp, delimiter=',')
                writer.writerows(step_data)

# ======================================================================================================================
    # Method that writes other information to file. It can be modified accordingly. Assumes that the input to be output
    #  has the shape (n,m), for example, (10,3).
    def create_special_output(self, step, ipn, ipn_pairs, w=None, dw=None):
        step_data = [['Part i', 'Part j', 'W', 'dW_x', 'dW_y', 'dW_z']]

        for pair in range(1, ipn + 1):

            new_data_line = [ipn_pairs[pair, 0], ipn_pairs[pair, 1], w[pair, 0], dw[pair, 0, 0], dw[pair, 0, 1],
                             dw[pair, 0, 2]]
            step_data.append(new_data_line)

        with open(self.path + 'Info' + str(step) + '.csv', 'wb') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerows(step_data)

# ======================================================================================================================
    # Method that writes the impact force on wall boundaries of type 2 to file. Assumes the input has shape (3,).
    @staticmethod
    def create_impact_output(file_name, force, time):
        with open(file_name, 'a') as csv_file:
            new_data_line = [[time, force[0], force[1], force[2]]]
            writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
            writer.writerows(new_data_line)

# ======================================================================================================================
