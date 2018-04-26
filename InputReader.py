# This class reads a CSV formatted input with the initial particle parameters and a TXT file with simulation parameters
#  and transforms them into a data structure and global variables that are used by the other classes of the SPH code.

__author__ = 'alomir'

import numpy as np


class InputReader:

    def __init__(self, csv_path, txt_path):
        self.path_csv = csv_path
        self.path_txt = txt_path
        self.sim_params = {}

    # Method that reads the input txt file and returns a dictionary which keys are the simulation parameters' and
    #   respective values.
    def read_txt(self):

        file_name = open(self.path_txt)

        # Read a line at a time and store the information as a list of strings
        for line in file_name:

            # Removes the end of line character.
            line = line.rstrip('\n')

            # Creates a list with the key and respective value.
            line_data = line.split('=')

            # The first element (position '0')of the list is the key value
            key = line_data[0]

            # The second element (position '1') of the list is the value
            value = line_data[1]
            self.sim_params[key] = value

        file_name.close()

        return self.sim_params

    # ==================================================================================================================
    # Method that reads the input csv file and returns arrays corresponding to each initial information of each
    #  particle.
    def read_csv(self):

        # Particle ID number
        # ID = np.genfromtxt(self.path_csv,delimiter=';',usecols=(0,),dtype=str)

        # Particle part_type
        part_type = np.genfromtxt(self.path_csv, delimiter=';', usecols=(1,), dtype=np.int)

        # Particle position
        r = np.genfromtxt(self.path_csv, delimiter=';', usecols=(2, 3, 4), dtype=np.float64)

        # Particle position
        v = np.genfromtxt(self.path_csv, delimiter=';', usecols=(5, 6, 7), dtype=np.float64)

        # Particle mass
        mass = np.genfromtxt(self.path_csv, delimiter=';', usecols=(9,), dtype=np.float64)

        # Particle density
        rho = np.genfromtxt(self.path_csv, delimiter=';', usecols=(10,), dtype=np.float64)

        # Particle internal energy
        e = np.genfromtxt(self.path_csv, delimiter=';', usecols=(11,), dtype=np.float64)

        # Body forces
        body = np.genfromtxt(self.path_csv, delimiter=';', usecols=(12, 13, 14), dtype=np.float64)

        # Initial stress state in Voight form
        stress = np.genfromtxt(self.path_csv, delimiter=';', usecols=(15, 16, 17, 18, 19, 20), dtype=np.float64)

        # Normals of the boundaries
        normals = np.genfromtxt(self.path_csv, delimiter=';', usecols=(21, 22, 23), dtype=np.float64)

        # Careful: do not use vectors here like r or v. They will give the correct size times three!
        size = np.size(part_type)

        # Full tensor
        sigma = np.zeros((size, 3, 3))
        sigma[1:, 0, 0] = stress[1:, 0]
        sigma[1:, 1, 1] = stress[1:, 1]
        sigma[1:, 2, 2] = stress[1:, 2]
        sigma[1:, 0, 1] = stress[1:, 3]
        sigma[1:, 1, 0] = stress[1:, 3]
        sigma[1:, 0, 2] = stress[1:, 4]
        sigma[1:, 2, 0] = stress[1:, 4]
        sigma[1:, 1, 2] = stress[1:, 5]
        sigma[1:, 2, 1] = stress[1:, 5]

        # Getting rid of the NaN at positions 0. This avoids warnings in the future.
        r[0, :] = 1e32
        v[0, :] = 0.0
        mass[0] = 0.0
        rho[0] = 1.0
        e[0] = 0.0
        body[0, :] = 0.0
        normals[0, :] = 0.0
        part_type[0] = -1

        return part_type, np.reshape(r, (size, 3)), np.reshape(v, (size, 3)), np.reshape(mass, (size, 1)), \
            np.reshape(rho, (size, 1)), np.reshape(e, (size, 1)), np.reshape(body, (size, 3)), sigma, \
            np.reshape(normals, (size, 3))
