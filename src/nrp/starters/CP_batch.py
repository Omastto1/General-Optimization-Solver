import os
import sys

from src.nrp.solvers.solver_cp import *
from src.general_optimization_solver import *


# File used to benchmark NRP on the cluster


def main(args):
    if len(args) != 3:
        print('Usage: python ...py <data_folder> <output_folder> <number of available cores>')
        print('Got:', args)
        return
    folder_path = args[0]
    output_folder = args[1]
    cores = int(args[2])

    # for folder_name in os.listdir(data_folder):
    if not os.path.isdir(folder_path):
        print('Usage: python ...py <data_folder> <output_folder> <number of available cores>')
        print('Got:', args)
        print('folder_path:', folder_path, 'is not a folder')
        return
    folder_name = os.path.basename(folder_path)

    print('folder_name:', folder_name)

    benchmark = load_benchmark(folder_path)

    solver = NRPSolver(TimeLimit=15*60, no_workers=cores)

    solver.solve(benchmark, validate=False, force_dump=True, output=output_folder)


if __name__ == "__main__":
    main(sys.argv[1:])
