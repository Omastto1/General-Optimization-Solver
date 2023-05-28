import json


def load_jobshop(path):
    with open(path, "r") as file:
        line = file.readline()

        while line.startswith("#"):
            line = file.readline()
        
        no_jobs, no_machines = [int(number.strip()) for number in line.split(" ")]

        machines = []
        durations = []
        for job in range(int(no_jobs)):
            line = file.readline()
            print(line.split(" "))
            numbers = [int(number.strip()) for number in line.split(" ") if len(number.strip())]
            job_machines = numbers[::2]
            job_durations = numbers[1::2]

            machines.append(job_machines)
            durations.append(job_durations)

        parsed_input = {
            "no_jobs": no_jobs,
            "no_machines": no_machines,
            "machines": machines,
            "durations": durations
        }
        
        return parsed_input


def load_jobshop_solution(file_path, instance):
    solution = {"feasible": False, "optimum": None}

    with open(file_path, "r") as file:
        instances_results = json.load(file)

        for instance_result in instances_results:
            if instance_result["name"] == instance:
                solution["feasible"] = True
                if instance_result["optimum"] is None:
                    solution["optimum"] = None
                    solution["bounds"] = instance_result["bounds"]
                else:
                    solution["optimum"] = instance_result["optimum"]

                break

    return solution
