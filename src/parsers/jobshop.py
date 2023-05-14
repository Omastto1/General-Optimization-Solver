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
