from docplex.cp.model import CpoModel
from collections import namedtuple

from .solver import Solver


class RCPSPSolver(Solver):
    def _solve_cp(self, instance, validate=False, visualize=False, force_execution=False):
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        model = CpoModel()
        model.set_parameters(params=self.params)

        x = [model.interval_var(size=duration, name=f"{i}") for i, duration in enumerate(
            instance.durations)]  # (4)

        model.add([model.minimize(model.max([model.end_of(x[i])
                for i in range(instance.no_jobs)]))])  # (1)

        model.add([model.sum(model.pulse(x[i], instance.requests[k][i]) for i in range(instance.no_jobs))
                <= instance.renewable_capacities[k] for k in range(instance.no_renewable_resources)])  # (2)

        model.add([model.end_before_start(x[i], x[successor - 1]) for (i, job_successors)
                in enumerate(instance.successors) for successor in job_successors])  # (3)

        sol = model.solve()

        if sol:
            if validate:
                try:
                    print("Validating solution...")
                    instance.validate(sol, x)
                    print("Solution is valid.")

                    obj_value = sol.get_objective_value()
                    print("Project completion time:", obj_value)
                        
                    instance.compare_to_reference(obj_value)
                except AssertionError as e:
                    print("Solution is invalid.")
                    print(e)
                    return None, None

            if visualize:
                instance.visualize(sol, x)


            # for i in range(no_jobs):
            #     print(f"Activity {i}: start={sol[i].get_start()}, end={sol[i].get_end()}")
        
            print(sol.solution.get_objective_bounds())
            print(sol.solution.get_objective_gaps())
            print(sol.solution.get_objective_values())

            # obj_value = sol.objective_value
            obj_value = sol.get_objective_values()[0]
            print('Objective value:', obj_value)
            instance.compare_to_reference(obj_value)
        else:
            print("No solution found.")        

        Solution = namedtuple("Solution", ['xs'])
        variables = Solution(x)

        instance.update_run_history(sol, variables, "CP", self.params)

        # print solution
        if sol.get_solve_status() == 'Optimal':
            print("Optimal solution found")
        elif sol.get_solve_status() == 'Feasible':
            print("Feasible solution found")
        else:
            print("Unknown solution status")
            print(sol.get_solve_status())


        return sol, variables


# def solve_rcpsp(no_jobs, no_resources, durations, successors, capacities, requests, validate=False):
#     mdl = CpoModel()

#     x = [ mdl.interval_var(size = duration, name=f"{i}") for i, duration in enumerate(durations) ] # (4)

#     mdl.add( [ mdl.minimize ( mdl.max( [mdl.end_of(x[i]) for i in range(no_jobs)] ) ) ] )# (1)

#     # mdl.add( [ mdl.sum( mdl.pulse(x[i],parsed_input["job_specifications"][i]["request_duration"][f"R{k+1}"]) for i in range(no_jobs)) <= parsed_input["resources"]["renewable_resources"]["renewable_availabilities"][k] for k in range(parsed_input["resources"]["renewable_resources"]["number_of_resources"]) ] )# (2)
#     mdl.add( [ mdl.sum( mdl.pulse(x[i], requests[k][i]) for i in range(no_jobs)) <= capacities[k] for k in range(no_resources) ] )# (2)

#     mdl.add( [mdl.end_before_start( x[i], x[successor - 1] ) for (i, job_successors) in enumerate(successors) for successor in job_successors] ) # (3)

#     # TODO: IS THIS NEEDED?
#     # Define the initial conditions
#     # mdl.add(x[0].get_start() == 0)

#     sol = mdl.solve(LogVerbosity='Terse')\
#     # sol = mdl.solve(TimeLimit=10)

#     if sol:
#         if validate:
#             try:
#                 print("Validating solution...")
#                 validate_rcpsp(sol, x, successors, sol.get_objective_value())
#                 print("Solution is valid.")
#             except AssertionError as e:
#                 print("Solution is invalid.")
#                 print(e)
#                 return None, None
            
#         print("Project completion time:", sol.get_objective_value())
#         # for i in range(no_jobs):
#         #     print(f"Activity {i}: start={sol[i].get_start()}, end={sol[i].get_end()}")
#     else:
#         print("No solution found.")

#     return sol, x


# def validate_rcpsp(sol, x, successors, horizon):
#     assert sol.get_objective_value() <= horizon, "Project completion time exceeds horizon."

#     for i, job_successors in enumerate(successors):
#         for successor in job_successors:
#             assert sol.get_var_solution(x[i]).get_end() <= sol.get_var_solution(x[successor - 1]).get_start(), f"Job {i} ends after job {successor} starts."

#     return True