import os
import sys

from src.vrp.solvers.ga_model import *
from src.general_optimization_solver import *

from pymoo.termination import get_termination
from pymoo.core.parameters import set_params, hierarchical
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA


# File used to benchmark VRP on the cluster


def main(args):
    if len(args) != 2:
        print('Usage: python CP.py <data_folder> <output_folder>')
        print('Got:', args)
        return
    folder_path = args[0]
    output_folder = args[1]
    # cores = int(args[2])

    # for folder_name in os.listdir(data_folder):
    if not os.path.isdir(folder_path):
        print('Usage: python CP.py <data_folder> <output_folder>')
        print('Got:', args)
        print('folder_path:', folder_path, 'is not a folder')
        return
    folder_name = os.path.basename(folder_path)

    print('folder_name:', folder_name)

    benchmark = load_benchmark(folder_path)  # , no_instances=2)
    termination = get_termination("time", "00:15:00")

    # algorithm = PSO()
    algorithm = DE()
    # algorithm = BRKGA()

    # hyperparams = {'mating.jitter': False, 'mating.CR': 0.4502132028215444, 'mating.crossover': 'bin', 'mating.F': 0.4277015784306394, 'mating.n_diffs': 1, 'mating.selection': 'rand'}
    # set_params(algorithm, hyperparams)

    solver = VRPTWSolver(algorithm=algorithm, fitness_func=fitness_func, termination=termination,
                         solver_name="GA DE")

    solver.solve(benchmark, validate=True, force_dump=True, output=output_folder)


if __name__ == "__main__":
    main(sys.argv[1:])
