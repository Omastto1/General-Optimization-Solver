

def print_verbose(print_input, verbose):
    if verbose:
        print(print_input)

# TODO: enable verbose mode
def load_j30(instance_path, verbose=False):
    parsed_input = {}

    with open(instance_path) as file:
        # ******
        print_verbose(file.readline(), verbose)
        # File with base data
        print_verbose(file.readline(), verbose)
        # initial value random generator
        print_verbose(file.readline(), verbose)
        # ******
        print_verbose(file.readline(), verbose)

        # projects : x
        no_projects = int(file.readline().split(":")[1].strip())
        parsed_input["no_projects"] = no_projects
        # jobs incl. supersource/sunk: y
        no_jobs = int(file.readline().split(":")[1].strip())
        parsed_input["number_of_jobs"] = no_jobs
        # horizon
        horizon = int(file.readline().split(":")[1].strip())
        parsed_input["horizon"] = horizon
        # Resources
        print_verbose(file.readline(), verbose)
        parsed_input["resources"] = {}
        # renewable
        no_renewable = int(file.readline().split(":")[1].split()[0].strip())
        parsed_input["resources"]["renewable_resources"] = {}
        parsed_input["resources"]["renewable_resources"]["number_of_resources"] = no_renewable

        # parsed_input["resources"]["no_renewable"] = no_renewable
        # non-renewable
        no_non_renewable = int(file.readline().split(":")[1].split()[0].strip())
        if no_non_renewable == 0:
            pass
        else:
            assert False, "no_non_renewable is non zero"
            # parsed_input["resources"]["no_non_renewable"] = no_non_renewable
        # doubly constrained
        no_doubly = int(file.readline().split(":")[1].split()[0].strip())
        if no_doubly == 0:
            pass
        else:
            assert False, "no_doubly is non zero"
            # parsed_input["resources"]["no_doubly_constrained"] = no_doubly

        # ******
        file.readline()
        # Project INFORMATION   
        parsed_input["project_information"] = {}
        file.readline()
        # pronr. #jobs rno_el. date, duedate, tardcost, MPM-Time   
        file.readline()
        project_info_line = file.readline()
        print_verbose("Project Information", verbose)
        print_verbose(project_info_line, verbose)
        pronr, no_jobs, rel_date, duedate, tardcost, mpm_time = [int(number.strip()) for number in project_info_line.split(" ") if len(number) > 0]
        parsed_input["project_information"]["pronr"] = pronr
        parsed_input["project_information"]["no_jobs"] = no_jobs
        parsed_input["project_information"]["rel_date"] = rel_date
        parsed_input["project_information"]["duedate"] = duedate
        parsed_input["project_information"]["tardcost"] = tardcost
        parsed_input["project_information"]["mpm_time"] = mpm_time
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
            line = [char.strip() for char in line.split(" ") if len(char) > 0]
            print_verbose(line, verbose)
            job_specification = {}

            job_nr = int(line[0].strip())
            job_specification["job_nr"] = job_nr
            no_modes = int(line[1].strip())
            job_specification["no_modes"] = no_modes
            no_successors = int(line[2].strip())
            job_specification["no_successors"] = no_successors
            successors = [int(node_no.strip()) for node_no in line[3:3+no_successors]]
            job_specification["successors"] = successors

            parsed_input["job_specifications"].append(job_specification)

            line = file.readline()

        # ******
        # file.readline()
        # requests / durations
        file.readline()
        # parsed_input["requests_durations"] = []
        # jobnr. mode duration R1 R2 R3 R4
        file.readline()
        # ---------
        file.readline()
        line = file.readline()
        job_no = 0
        while not line.startswith("*"):
            line = [char.strip() for char in line.split(" ") if len(char) > 0]
            print_verbose(line, verbose)
            # request_duration = {}
            project_specification = parsed_input["job_specifications"][job_no]

            job_nr = int(line[0].strip())
            print_verbose(f"{job_nr}, {project_specification['job_nr']}", verbose)
            assert job_nr == project_specification["job_nr"]
            # request_duration["job_nr"] = job_nr
            mode = int(line[1].strip())
            assert mode == project_specification["no_modes"]
            # request_duration["mode"] = mode

            mode_dict = {}
            mode_dict["mode"] = mode
            duration = int(line[2].strip())
            mode_dict["duration"] = duration
            # request_duration["duration"] = duration
            # project_specification["duration"] = duration
            resource_req = [int(req.strip()) for req in line[3:7]]

            mode_dict["request_duration"] = {}
            mode_dict["request_duration"]["R1"] = resource_req[0]
            mode_dict["request_duration"]["R2"] = resource_req[1]
            mode_dict["request_duration"]["R3"] = resource_req[2]
            mode_dict["request_duration"]["R4"] = resource_req[3]

            project_specification["modes"] = [mode_dict]

            parsed_input["job_specifications"][job_no] = project_specification

            print_verbose(f"job {job_no} done, {project_specification}", verbose)

            # parsed_input["requests_durations"].append(request_duration)

            line = file.readline()
            job_no += 1
        
        
        # ******
        # file.readline()
        # resource availabilities
        file.readline()
        parsed_input["resource_availabilities"] = {}
        #  R1 R2 R3 R4
        file.readline()
        line = [int(number.strip()) for number in file.readline().split(" ") if len(number) > 0]
        print_verbose(line, verbose)
        parsed_input["resource_availabilities"]["R1"] = line[0]
        parsed_input["resource_availabilities"]["R2"] = line[1]
        parsed_input["resource_availabilities"]["R3"] = line[2]
        parsed_input["resource_availabilities"]["R4"] = line[3]
        
        parsed_input["resources"]["renewable_resources"]["renewable_availabilities"] = line
    
    return parsed_input


def load_j30_solution(file_path, benchmark_name, instance):
    # parameter number is placed directly after the benchmark name in the instance name
    # e.g. j301_1.sm -> benchmark_name = "j30", parameter = 1, instance = 1
    parameter = instance.split("_")[0][len(benchmark_name):]
    instance = instance.split("_")[1].split(".")[0]
    solution = {"feasible": False, "optimum": None, "cpu_time": None}
    print(f"Loading solution for {benchmark_name} {parameter} {instance} ...")

    with open(file_path, "r") as file:
        line = ""

        row = -1
        while not line.startswith("-----"):
            line = file.readline()
            row += 1

            if row > 100000:
                raise Exception("Did not encounter `------` line separating solution from header, cancelling ...")
            
        while line != "":
            line = [char.strip() for char in line.split() if len(char.strip()) > 0]

            if line[0] == parameter and line[1] == instance:
                if file_path.split(".")[-2].endswith("opt"):
                    solution = {"feasible": True, "optimum": int(line[2]), "cpu_time": float(line[3])}
                elif file_path.split(".")[-2].endswith("lb"):
                    upper_bound = int(line[2])
                    lower_bound = int(line[3])

                    if upper_bound == lower_bound:  # optimum is known
                        solution = {"feasible": True, "optimum": lower_bound}
                    else:
                        solution = {"feasible": True, "optimum": None, "bounds":{"upper": int(line[2]), "lower": int(line[3])}}
                else:
                    raise Exception("Unknown solution format")

                break
            
            line = file.readline()

    return solution
