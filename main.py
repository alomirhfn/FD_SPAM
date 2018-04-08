from __future__ import print_function
import InputReader
import OutputModule
import Solver

from Timer import Timer

__author__ = 'alomir'

# outPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Moriguchi/'
# CSVPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Moriguchi.csv'
# TXTPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Moriguchi.txt'

outPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Test/'
CSVPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Plane_Strain.csv'
TXTPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Plane_Strain.txt'

# outPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Slump_Lube_3D/'
# CSVPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Slump_Lube_3D.csv'
# TXTPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Slump_Lube_3D.txt'

# outPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Pudasaini_05_3D/'
# CSVPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Pudasaini_05_3D.csv'
# TXTPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Pudasaini_05_3D.txt'

# outPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Lube/Cython/L2_Debug/'
# CSVPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Lube_2.csv'
# TXTPath = 'C:/Users/alomir/Documents/Doutorado/Stanford/Thesis/SPH_CODE/SPAM_IO/Lube_phi_37_debug.txt'

# PROFILING = True
PROFILING = False


def main():

    t = Timer()

    input_object = InputReader.InputReader(CSVPath, TXTPath)

    t.tic()
    part_type, init_r, init_v, init_temp, mass, init_rho, init_press, init_energy, body_forces, init_sigma, normals = \
        input_object.read_csv()
    params_list = input_object.read_txt()
    print()
    t.toc("Loading simulation data")

    t.tic()
    output_object = OutputModule.Output(outPath)
    solver_object = Solver.Solver(params_list, init_r)
    solver_object.solve(part_type, init_r, init_v, init_rho, mass, init_sigma, init_energy, body_forces, output_object)
    t.toc("Problem solution")


if __name__ == '__main__':
    if PROFILING:
        import cProfile
        cProfile.run('main()', sort='time')
    else:
        main()
