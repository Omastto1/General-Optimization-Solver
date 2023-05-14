from docplex.cp.model import CpoModel
from src.optimization_problem import OptimizationProblem


class RCPSP(OptimizationProblem):
    def __init__(self, benchmark_name, instance_name, format_, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, format_, data, solution, run_history)

        self.no_jobs = self._data["number_of_jobs"]  # number of activities
        self.no_renewable_resources = self._data["resources"]["renewable_resources"]["number_of_resources"]
        self.horizon = self._data["horizon"] if "horizon" in self._data else 2^32
        self.durations = [job["modes"][0]["duration"] for job in self._data["job_specifications"]]  # duration of each activity
        self.successors = [job["successors"] for job in self._data["job_specifications"]] # # precedence constraints
        self.renewable_capacities = self._data["resources"]["renewable_resources"]["renewable_availabilities"]  # available resource capacity
        self.requests = [[self._data["job_specifications"][i]["modes"][0]["request_duration"][f"R{k+1}"] for i in range(self.no_jobs)] for k in range(self.no_renewable_resources) ]
# [resource_request for resource_request in mode["request_duration"].values()]

    def solve(self, validate=False):
        mdl = CpoModel()

        x = [ mdl.interval_var(size = duration, name=f"{i}") for i, duration in enumerate(self.durations) ] # (4)

        mdl.add( [ mdl.minimize ( mdl.max( [mdl.end_of(x[i]) for i in range(self.no_jobs)] ) ) ] )# (1)

        # mdl.add( [ mdl.sum( mdl.pulse(x[i],parsed_input["job_specifications"][i]["request_duration"][f"R{k+1}"]) for i in range(no_jobs)) <= parsed_input["resources"]["renewable_resources"]["renewable_availabilities"][k] for k in range(parsed_input["resources"]["renewable_resources"]["number_of_resources"]) ] )# (2)
        mdl.add( [ mdl.sum( mdl.pulse(x[i], self.requests[k][i]) for i in range(self.no_jobs)) <= self.renewable_capacities[k] for k in range(self.no_renewable_resources) ] )# (2)

        mdl.add( [mdl.end_before_start( x[i], x[successor - 1] ) for (i, job_successors) in enumerate(self.successors) for successor in job_successors] ) # (3)

        # TODO: IS THIS NEEDED?
        # Define the initial conditions
        # mdl.add(x[0].get_start() == 0)

        sol = mdl.solve(LogVerbosity='Terse')\
        # sol = mdl.solve(TimeLimit=10)

        if sol:
            if validate:
                try:
                    print("Validating solution...")
                    self.validate(sol, x)
                    print("Solution is valid.")
                except AssertionError as e:
                    print("Solution is invalid.")
                    print(e)
                    return None, None
                
            print("Project completion time:", sol.get_objective_value())
            # for i in range(no_jobs):
            #     print(f"Activity {i}: start={sol[i].get_start()}, end={sol[i].get_end()}")
        else:
            print("No solution found.")

        return sol, x
    
    def validate(self, sol, x):
        assert sol.get_objective_value() <= self.horizon, "Project completion time exceeds horizon."

        for i, job_successors in enumerate(self.successors):
            for successor in job_successors:
                assert sol.get_var_solution(x[i]).get_end() <= sol.get_var_solution(x[successor - 1]).get_start(), f"Job {i} ends after job {successor} starts."

        return True
