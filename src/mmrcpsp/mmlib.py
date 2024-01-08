import pandas as pd


def print_verbose(print_input, verbose):
    if verbose:
        print(print_input)


def load_mmlib(instance_path, verbose=False):
    parsed_input = {}

    with open(instance_path) as file:
        # jobs incl. supersource/sunk: y
        no_jobs = int(file.readline().split(":")[1].strip())
        parsed_input["number_of_jobs"] = no_jobs

        # Resources
        print_verbose(file.readline(), verbose)
        parsed_input["resources"] = {}

        # renewable
        no_renewable = int(file.readline().split(":")[1].split()[0].strip())
        parsed_input["resources"]["renewable_resources"] = {}
        parsed_input["resources"]["renewable_resources"]["number_of_resources"] = no_renewable

        # non-renewable
        no_non_renewable = int(file.readline().split(":")[
                               1].split()[0].strip())
        parsed_input["resources"]["non_renewable_resources"] = {}
        parsed_input["resources"]["non_renewable_resources"]["number_of_resources"] = no_non_renewable

        # doubly constrained
        no_doubly = int(file.readline().split(":")[1].split()[0].strip())
        parsed_input["resources"]["doubly_constrained_resources"] = {}
        parsed_input["resources"]["doubly_constrained_resources"]["number_of_resources"] = no_doubly

        # ******
        file.readline()

        # Precedence relation
        file.readline()
        parsed_input["job_specifications"] = []

        # parsed_input["precedence_relations"] = []
        # jobnr. # modes #successorcs successors
        print_verbose("precedence relations", verbose)
        file.readline()
        line = file.readline()

        while not line.startswith("*"):
            line = [char.strip() for char in line.split() if len(char) > 0]
            print_verbose(line, verbose)
            job_specification = {}

            job_nr = int(line[0].strip())
            job_specification["job_nr"] = job_nr
            no_modes = int(line[1].strip())
            job_specification["no_modes"] = no_modes
            job_specification["modes"] = []
            no_successors = int(line[2].strip())
            job_specification["no_successors"] = no_successors
            successors = [int(node_no.strip())
                          for node_no in line[3:3+no_successors]]
            job_specification["successors"] = successors

            parsed_input["job_specifications"].append(job_specification)

            line = file.readline()

        # requests / durations
        file.readline()

        # parsed_input["requests_durations"] = []
        # jobnr. mode duration R1 R2 R3 R4
        file.readline()

        # ---------
        file.readline()
        line = file.readline()

        while not line.startswith("*"):
            line = [char.strip()
                    for char in line.split() if len(char.strip()) > 0]
            print_verbose(line, verbose)

            if len(line) == 7:
                job_nr = int(line[0].strip())

            project_specification = parsed_input["job_specifications"][job_nr - 1]

            print_verbose(
                f"{job_nr}, {project_specification['job_nr']}", verbose)

            # assert job_nr == project_specification["job_nr"]
            # request_duration["job_nr"] = job_nr
            if len(line) == 7:
                mode = int(line[1].strip())
            else:
                mode = int(line[0].strip())
            print_verbose(f"mode {mode}", verbose)
            assert mode <= project_specification["no_modes"]

            # request_duration["mode"] = mode
            if len(line) == 7:
                duration = int(line[2].strip())
            else:
                duration = int(line[1].strip())

            mode_description = {}

            mode_description["mode"] = mode
            mode_description["duration"] = duration

            if len(line) == 7:
                resource_req = [int(req.strip()) for req in line[3:7]]
            else:
                resource_req = [int(req.strip()) for req in line[-4:]]

            print_verbose(resource_req, verbose)
            mode_description["request_duration"] = {}
            mode_description["request_duration"]["R1"] = resource_req[0]
            mode_description["request_duration"]["R2"] = resource_req[1]
            mode_description["request_duration"]["N1"] = resource_req[2]
            mode_description["request_duration"]["N2"] = resource_req[3]

            project_specification["modes"].append(mode_description)

            parsed_input["job_specifications"][job_nr -
                                               1] = project_specification

            print_verbose(
                f"job {job_nr} done, {project_specification}", verbose)

            line = file.readline()

        # resource availabilities
        file.readline()

        # empty line
        file.readline()
        parsed_input["resource_availabilities"] = {}

        #  R1 R2 R3 R4
        file.readline()
        line = [int(number.strip())
                for number in file.readline().split() if len(number) > 0]
        print_verbose(line, verbose)
        parsed_input["resource_availabilities"]["R1"] = line[0]
        parsed_input["resource_availabilities"]["R2"] = line[1]
        parsed_input["resource_availabilities"]["N1"] = line[2]
        parsed_input["resource_availabilities"]["N2"] = line[3]

        parsed_input["resources"]["renewable_resources"]["renewable_availabilities"] = line[:2]
        parsed_input["resources"]["non_renewable_resources"]["non_renewable_availabilities"] = line[2:]

    return parsed_input


def load_mmlib_solution(file_path, instance_name):
    solution = {"feasible": False, "optimum": None}

    instances_results = pd.read_excel(file_path)

    instance_result = instances_results[instances_results["Instance"]
                                        == instance_name]

    assert instance_result.shape[
        0] == 1, f"Found zero or two and more reference results with the {instance_name} instance name"
    instance_result = instance_result.iloc[0]

    lower_bound = int(instance_result["Best LB of our approach"])
    upper_bound = max(int(instance_result["Best known makespan"]), int(
        instance_result["Best found makespan"]))

    if upper_bound == lower_bound:
        solution = {"feasible": True, "optimum": int(upper_bound)}
    else:
        solution = {"feasible": True, "optimum": None, "bounds": {
            "upper": upper_bound, "lower": lower_bound}}

    return solution
