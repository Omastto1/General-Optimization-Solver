import pandas as pd


# TODO: enable verbose mode
def load_patterson(instance_path, verbose=False):
    # the patterson format
    parsed_input = {}

    with open(instance_path) as file:
        # 1. line: blank line
        line = file.readline()
        while len(line.strip()) == 0:
            line = file.readline()
        print(line)
        # 2. line: Number of activities (starting with node 1 and two dummy nodes inclusive), Number of renewable resource
        number_of_jobs, number_of_renewable_resources = [int(number.strip()) for number in line.split(" ") if len(number.strip()) > 0]
        parsed_input["number_of_jobs"] = number_of_jobs
        parsed_input["resources"] = {}
        # parsed_input["number_of_renewable_resources"] = number_of_renewable_resources
        # parsed_input["resources"]["no_non_renewable"] = 0
        # parsed_input["resources"]["no_doubly_constrained"] = 0

        # 3. line: Availability for each renewable resource
        resource_availabilities = [int(number.strip()) for number in file.readline().split(" ") if len(number.strip()) > 0]
        
        parsed_input["resources"]["renewable_resources"] = {}
        parsed_input["resources"]["renewable_resources"]["number_of_resources"] = number_of_renewable_resources
        parsed_input["resources"]["renewable_resources"]["renewable_availabilities"] = resource_availabilities
        
        # 4. line: blank line
        line = file.readline()
        while len(line.strip()) == 0:
            line = file.readline()
        print(line)

        parsed_input["job_specifications"] = []
        # following lines
        # duration, renewable resource consumption, number of successors, successors
        job_no = 0
        line = [int(number.strip()) for number in line.split(" ") if len(number.strip()) > 0]
        ending_line = [0] + [0] * number_of_renewable_resources + [0]  # duration, renewable resources consumption, number of succesors
        while line != ending_line :
            print(line)
            job_specification = {}

            job_specification["job_nr"] = job_no
            job_specification["no_modes"] = 1

            duration = line[0]
            renewable_resource_consumption = line[1:1+int(number_of_renewable_resources)]

            mode = {}
            mode["mode"] = 1
            mode["duration"] = duration
            mode["request_duration"] = {}

            for resource_index in range(number_of_renewable_resources):
                mode["request_duration"][f"R{resource_index+1}"] = renewable_resource_consumption[resource_index]
            
            job_specification["modes"] = [mode]
            
            number_of_successors = line[1+int(number_of_renewable_resources)]
            job_specification["number_of_successors"] = number_of_successors

            successors = line[2+int(number_of_renewable_resources):2+int(number_of_renewable_resources)+int(number_of_successors)]

            if len(successors) < number_of_successors:
                line = file.readline()
                while line.strip() == "":
                    line = file.readline()
                
                line = [int(number.strip()) for number in line.split(" ") if len(number.strip()) > 0]

                assert len(line) == number_of_successors - len(successors), "Number of successors does not match the number of successors given in the line"

                successors += line
                
            job_specification["successors"] = successors

            
            line = file.readline()
            while line.strip() == "":
                line = file.readline()
            
            line = [int(number.strip()) for number in line.split(" ") if len(number.strip()) > 0]

            parsed_input["job_specifications"].append(job_specification)

            job_no += 1

    terminal_node_resources_consumption = {f"R{resource_index}": 0 for resource_index in range(1, number_of_renewable_resources+1)}
    job_specification = {"job_nr": number_of_jobs, "no_modes": 1, "modes": [{"mode": 1, "duration": 0, "request_duration": terminal_node_resources_consumption}], "number_of_successors": 0, "successors": []}
    parsed_input["job_specifications"].append(job_specification)
    parsed_input

    return parsed_input


def load_patterson_solution(file_path, instance_name):
    solution = {"feasible": False, "optimum": None}
    print(instance_name)
    print("aaaaa")

    instances_results = pd.read_excel(file_path)

    print(instances_results[instances_results["Instance name"] == instance_name])
    instance_result = instances_results[instances_results["Instance name"] == instance_name].iloc[0]


    assert instance_result.shape == (3,), "Instance reference result not found"

    upper_bound = int(instance_result["UB-lit"])
    lower_bound = int(instance_result["LB-lit"])

    if upper_bound == lower_bound:
        solution = {"feasible": True, "optimum": int(upper_bound)}
    else:
        solution = {"feasible": True, "optimum": None, "bounds":{"upper": upper_bound, "lower": lower_bound}}

    return solution
