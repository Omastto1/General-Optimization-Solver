from docplex.cp.model import CpoModel
from src.rcpsp import RCPSP


class MMRCPSP(RCPSP):
    def __init__(self, benchmark_name, instance_name, format_, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, format_, data, solution, run_history)

        # new
        self.no_non_renewable_resources = self._data["resources"]["non_renewable_resources"]["number_of_resources"]  # number of non-renewable resources
        self.non_renewable_capacities = self._data["resources"]["non_renewable_resources"]["non_renewable_availabilities"]  # available resource capacity
        self.no_modes_list = [job_specification['no_modes'] for job_specification in self._data['job_specifications']]


        resource_keys = ["R1", "R2", "N1", "N2"]
        # overwritte rcpsp variables
        self.durations = [[mode["duration"] for mode in job["modes"]] for job in self._data["job_specifications"] ]  # duration of each activity
        self.requests = [[[mode["request_duration"][k] for mode in job["modes"]] for job in self._data["job_specifications"]  ] for k in resource_keys]


    def solve(self, validate=False):
        # define model
        model = CpoModel()

        # define variables
        tasks = range(self.no_jobs)
        renewable_resources = range(self.no_renewable_resources)
        non_renewable_resources = range(self.no_non_renewable_resources)

        xs = [model.interval_var(name=f'task_{i}') for i in tasks]
        ys = [[model.interval_var(size=self.durations[i][j], name=f'task_{i}_mode_{j}', optional=True) for j in range(self.no_modes_list[i])] for i in tasks]



        model.add(model.minimize(model.max(model.end_of(x) for x in xs)))

        for i in tasks:
            model.add(model.alternative(xs[i], ys[i]))

        for k in renewable_resources:
            renewable_resources_requirements = [model.pulse(ys[i][j], self.requests[k][i][j]) for i in tasks for j in range(self.no_modes_list[i])]
            model.add(model.sum(renewable_resources_requirements) <= self.renewable_capacities[k])

        for k in non_renewable_resources:
            non_renewable_resources_requirements = [model.presence_of(ys[i][j]) * self.requests[k+2][i][j] for i in tasks for j in range(self.no_modes_list[i])]
            model.add(model.sum(non_renewable_resources_requirements) <= self.non_renewable_capacities[k])

        # define precedence constraints
        for (i, job_successors) in enumerate(self.successors):
            model.add([model.end_before_start( xs[i], xs[successor - 1] )  for successor in job_successors])

        # solve model
        sol = model.solve()

        if sol.get_solve_status() in ["Unknown", "Infeasible", "JobFailed", "JobAborted"]:
            print('No solution found')

        if validate:
            try:
                print("Validating solution...")
                validate(sol, xs, self.successors, sol.get_objective_value())
                print("Solution is valid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, None

        # print solution
        if sol.get_solve_status() == 'Optimal':
            print("Optimal solution found")
        elif sol.get_solve_status() == 'Feasible':
            print("Feasible solution found")
        else:
            print("Unknown solution status")
            print(sol.get_solve_status())

        print('Objective value:', sol.get_objective_value())
        for i in tasks:
            print(sol.get_var_solution(xs[i]))
            for j in range(self.no_modes_list[i]):
                if sol.get_var_solution(ys[i][j]).is_absent():
                    continue
                
                print(f'Task {i} is scheduled in mode {j} from {sol.get_var_solution(ys[i][j]).start} to {sol.get_var_solution(ys[i][j]).end}')
                # if sol.get_var_solution(y[i][j]) == 1:
                #     print(f'Task {i} is scheduled in mode {j} from {sol.get_var_solution(x[i][j]).start} to {sol.get_var_solution(x[i][j]).end}')
        # for k in resources:
        #     print(f'Resource {k} usage: {sol.get_var_solution(z[k])}')

        xs = [ys[i][j] for i in tasks for j in range(self.no_modes_list[i]) if sol.get_var_solution(ys[i][j]).is_present()]

        return sol, xs
    
    def validate(self, sol, x):
        super().validate(sol, x)