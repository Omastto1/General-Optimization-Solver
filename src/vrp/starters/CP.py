from src.vrp.solvers.interval_model import *
# from integer_model import *
# from ORsolver import *


# File used to benchmark VRP on the cluster


def main(args):
    if len(args) != 3:
        print('Usage: python CP.py <data_folder> <output_folder> <number of available cores>')
        print('Got:', args)
        return
    folder_path = args[0]
    cores = int(args[2])

    # for folder_name in os.listdir(data_folder):
    if not os.path.isdir(folder_path):
        print('Usage: python CP.py <data_folder> <output_folder> <number of available cores>')
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

        tlim = 15 * 60

        print('instance_name:', instance_name)

        instance = IntervalModel()
        instance.read_json(instance_path)

        if instance.instance['best_known_solution'] is not None:
            print('best_known_solution:', instance.instance['best_known_solution']['Distance'])
        else:
            print('best_known_solution: None')

        instance.solve(tlim, workers=cores)

        print('solution:', instance.solution['total_distance'])

        output = os.path.join(args[1], instance_name + '.json')
        if not os.path.exists(os.path.dirname(output)):
            os.makedirs(os.path.dirname(output))
        print('Saving to', output)
        instance.save_to_json(output)

        return


if __name__ == "__main__":
    main(sys.argv[1:])

# "C:\Users\micha\OneDrive - České vysoké učení technické v Praze\Dokumenty\PycharmProjects\optimizin\General-Optimization-Solver\data\CVRPTW\solomon_25"
# "C:\Users\micha\OneDrive - České vysoké učení technické v Praze\Dokumenty\PycharmProjects\optimizin\General-Optimization-Solver\data\out\solomon_25"
# 6
