import os
from cvrptw import *


def main(args):
    if len(args) != 3:
        print('Usage: python starter.py <data_folder> <output_folder> <number of available cores>')
        return
    data_folder = args[0]
    cores = args[3]

    for folder_name in os.listdir(data_folder):
        if not os.path.isdir(os.path.join(data_folder, folder_name)):
            continue
        N = int(folder_name.split('_')[-1])  # Extract the value of N from the folder name
        folder_path = os.path.join(data_folder, folder_name)
        print('folder_name:', folder_name)
        for filename in os.listdir(folder_path):
            if not filename.endswith('.json'):
                continue
            instance_name = filename.split('.')[0]  # Extract the instance name from the file name
            instance_path = os.path.join(folder_path, filename)

            tlim = 60*1  # 15 minutes

            print('instance_name:', instance_name)

            instance = Cvrptw()
            instance.read_json(instance_path)

            print('best_known_solution:', instance.instance['best_known_solution']['Distance'])

            instance.solve(tlim, workers=cores)

            print('solution:', instance.solution['total_distance'])

            output = os.path.join(args[1], folder_name, instance_name + '.json')
            if not os.path.exists(os.path.dirname(output)):
                os.makedirs(os.path.dirname(output))
            print('Saving to', output)
            instance.save_to_json(output)

            return


if __name__ == "__main__":
    main(sys.argv[1:])
