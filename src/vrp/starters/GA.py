import os
import sys

from src.vrp.solvers.ga_model import *

from pymoo.termination import get_termination

# File used to benchmark VRP on the cluster


def main(args):
    if len(args) != 2:
        print('Usage: python CP.py <data_folder> <output_folder>')
        print('Got1:', args)
        return
    folder_path = args[0]
    # cores = int(args[2])

    termination = get_termination("time", "00:00:10")

    # for folder_name in os.listdir(data_folder):
    if not os.path.isdir(folder_path):
        print('Usage: python CP.py <data_folder> <output_folder>')
        print('Got:', args)
        return
    folder_name = os.path.basename(folder_path)
    # N = int(folder_name.split('_')[-1])  # Extract the value of N from the folder name
    # folder_path = os.path.join(data_folder, folder_name)
    print('folder_name:', folder_name)
    for filename in os.listdir(folder_path):
        print('filename:', filename)
        if not filename.endswith('.json'):
            continue
        # folder_name = 'solomon_25'
        # filename = 'R112.json'
        instance_name = filename.split('.')[0]  # Extract the instance name from the file name
        instance_path = os.path.join(folder_path, filename)

        print('instance_name:', instance_name)

        termination = get_termination("time", "00:15:00")
        algorithm = PSO()
        instance = load_instance(instance_path)
        solver = VRPTWSolver(algorithm=algorithm, fitness_func=fitness_func, termination=termination,
                             solver_name="GA PSO")

        res = solver.solve(instance, validate=True, visualize=True)

        print('solution:', res[0])

        # output = os.path.join(args[1], instance_name + '.json')
        # if not os.path.exists(os.path.dirname(output)):
        #     os.makedirs(os.path.dirname(output))
        # print('Saving to', output)
        instance.dump(dir_path=args[1])

        return


if __name__ == "__main__":
    main(sys.argv[1:])
