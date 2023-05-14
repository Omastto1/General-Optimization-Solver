

# TODO: enable verbose mode
def load_patterson(instance_path, verbose=False):
    # the patterson format
    parsed_input = {}

    with open(instance_path) as file:
        # 1. line: blank line
        file.readline()
        # 2. line: Number of activities (starting with node 1 and two dummy nodes inclusive), Number of renewable resource
        number_of_jobs, number_of_renewable_resources = [int(number.strip()) for number in file.readline().split(" ") if len(number.strip()) > 0]
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
        file.readline()

        parsed_input["job_specifications"] = []
        # following lines
        # duration, renewable resource consumption, number of successors, successors
        job_no = 0
        line = [int(number.strip()) for number in file.readline().split(" ") if len(number.strip()) > 0]
        while line != [0, 0, 0, 0, 0, 0]:
            print(line)
            job_specification = {}

            job_specification["job_nr"] = job_no
            job_specification["no_modes"] = 1

            duration = line[0]
            renewable_resource_consumption = line[1:1+int(number_of_renewable_resources)]

            mode = {}
            mode["mode_nr"] = 1
            mode["duration"] = duration
            mode["request_duration"] = {}
            mode["request_duration"]["R1"] = renewable_resource_consumption[0]
            mode["request_duration"]["R2"] = renewable_resource_consumption[1]
            mode["request_duration"]["R3"] = renewable_resource_consumption[2]
            mode["request_duration"]["R4"] = renewable_resource_consumption[3]

            job_specification["modes"] = [mode]
            
            number_of_successors = line[1+int(number_of_renewable_resources)]
            job_specification["number_of_successors"] = number_of_successors

            successors = line[2+int(number_of_renewable_resources):2+int(number_of_renewable_resources)+int(number_of_successors)]
            job_specification["successors"] = successors
            
            line = [int(number.strip()) for number in file.readline().split(" ") if len(number.strip()) > 0]

            parsed_input["job_specifications"].append(job_specification)

    job_specification = {"job_nr": number_of_jobs, "duration": 0, "request_duration": {"R1": 0, "R2": 0, "R3": 0, "R4": 0}, "number_of_successors": 0, "successors": []}
    parsed_input["job_specifications"].append(job_specification)
    parsed_input

    return parsed_input
