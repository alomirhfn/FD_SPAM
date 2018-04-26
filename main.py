from __future__ import print_function
import InputReader
import OutputModule
import Solver

from Timer import Timer

__author__ = 'alomir'

# outPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Test/'
# CSVPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Plane_Strain.csv'
# TXTPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Plane_Strain.txt'

outPath = 'C:/Users/alomir/Desktop/Output/Art_Visc/'
CSVPath = 'C:/Users/alomir/Desktop/SPAM_IO/2D_Colapse_Nguyen_16.csv'
TXTPath = 'C:/Users/alomir/Desktop/SPAM_IO/2D_Colapse_Nguyen_16.txt'

# Provides all statistics in terms of running time for each part of the code after it is done. Use it to identify
#  bottlenecks and optimization.
PROFILING = False


def main():

    t = Timer()

    input_object = InputReader.InputReader(CSVPath, TXTPath)

    t.tic()
    part_type, init_r, init_v, mass, init_rho, init_energy, body_forces, init_sigma, normals = input_object.read_csv()
    params_list = input_object.read_txt()
    print()
    t.toc("Loading simulation data")

    t.tic()
    output_object = OutputModule.Output(outPath)
    solver_object = Solver.Solver(params_list, init_r, init_rho)
    solver_object.solve(part_type, init_r, init_v, init_rho, mass, init_sigma, init_energy, body_forces, output_object)
    t.toc("Problem solution")


if __name__ == '__main__':
    if PROFILING:
        import cProfile
        cProfile.run('main()', sort='time')
    else:
        main()
