from docplex.cp.model import CpoModel
from collections import namedtuple

from .solver import Solver


class MMRCPSPSolver(Solver):
    def _solve_cp(self, instance, validate=False, visualize=False, force_execution=False):
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        # define model
        model = CpoModel()
        model.set_parameters(params=self.params)

        # define variables
        tasks = range(instance.no_jobs)
        renewable_resources = range(instance.no_renewable_resources)
        non_renewable_resources = range(instance.no_non_renewable_resources)

        xs = [model.interval_var(name=f'task_{i}') for i in tasks]
        ys = [[model.interval_var(size=instance.durations[i][j], name=f'task_{i}_mode_{j}', optional=True) for j in range(
            instance.no_modes_list[i])] for i in tasks]

        cost = model.integer_var(0, 1000000, name="cost")

        model.add(cost == model.max(model.end_of(x) for x in xs))

        if "optimum" in instance._solution and instance._solution["optimum"] is not None:
            model.add(cost >= instance._solution["optimum"])

        model.add(model.minimize(model.max(model.end_of(x) for x in xs)))

        for i in tasks:
            model.add(model.alternative(xs[i], ys[i]))

        for k in renewable_resources:
            renewable_resources_requirements = [model.pulse(
                ys[i][j], instance.requests[k][i][j]) for i in tasks for j in range(instance.no_modes_list[i])]
            model.add(model.sum(renewable_resources_requirements)
                      <= instance.renewable_capacities[k])

        for k in non_renewable_resources:
            non_renewable_resources_requirements = [model.presence_of(
                ys[i][j]) * instance.requests[k+2][i][j] for i in tasks for j in range(instance.no_modes_list[i])]
            model.add(model.sum(non_renewable_resources_requirements)
                      <= instance.non_renewable_capacities[k])

        # define precedence constraints
        for (i, job_successors) in enumerate(instance.successors):
            model.add([model.end_before_start(xs[i], xs[successor - 1])
                      for successor in job_successors])

        # solve model
        sol = model.solve()

        if sol.get_solve_status() in ["Unknown", "Infeasible", "JobFailed", "JobAborted"]:
            print('No solution found')

        if validate:
            try:
                print("Validating solution...")
                instance.validate(sol, xs)
                print("Solution is valid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, None

        if visualize:
            instance.visualize(sol, xs, ys)
        
        for i in tasks:
            print(sol.get_var_solution(xs[i]))
            for j in range(instance.no_modes_list[i]):
                if sol.get_var_solution(ys[i][j]).is_absent():
                    continue

                print(
                    f'Task {i} is scheduled in mode {j} from {sol.get_var_solution(ys[i][j]).start} to {sol.get_var_solution(ys[i][j]).end}')
                # if sol.get_var_solution(y[i][j]) == 1:
                #     print(f'Task {i} is scheduled in mode {j} from {sol.get_var_solution(x[i][j]).start} to {sol.get_var_solution(x[i][j]).end}')
        # for k in resources:
        #     print(f'Resource {k} usage: {sol.get_var_solution(z[k])}')
        
        # print solution
        if sol.get_solve_status() == 'Optimal':
            print("Optimal solution found")
        elif sol.get_solve_status() == 'Feasible':
            print("Feasible solution found")
        else:
            print("Unknown solution status")
            print(sol.get_solve_status())

        # obj_value = sol.get_objective_value()
        # obj_value = sol.get_objective_value()

        obj_value = sol.get_objective_values()[0]
        print('Objective value:', obj_value)
        instance.compare_to_reference(obj_value)

        Solution = namedtuple("Solution", ['xs'])
        
        xs = [ys[i][j] for i in tasks for j in range(
            instance.no_modes_list[i]) if sol.get_var_solution(ys[i][j]).is_present()]
        variables = Solution(xs)

        instance.update_run_history(sol, variables, "CP", self.params)

        return sol, variables

# def solve_mmrcpsp(no_jobs, no_modes_list, no_renewable_resources, no_non_renewable_resources, durations, successors, renewable_capacities, non_renewable_capacities, requests, validate=False):
#     # define model
#     model = CpoModel()

#     # define variables
#     tasks = range(no_jobs)
#     renewable_resources = range(no_renewable_resources)
#     non_renewable_resources = range(no_non_renewable_resources)

#     xs = [model.interval_var(name=f'task_{i}') for i in tasks]
#     ys = [[model.interval_var(size=durations[i][j], name=f'task_{i}_mode_{j}', optional=True) for j in range(no_modes_list[i])] for i in tasks]



#     model.add(model.minimize(model.max(model.end_of(x) for x in xs)))

#     for i in tasks:
#         model.add(model.alternative(xs[i], ys[i]))

#     for k in renewable_resources:
#         renewable_resources_requirements = [model.pulse(ys[i][j], requests[k][i][j]) for i in tasks for j in range(no_modes_list[i])]
#         model.add(model.sum(renewable_resources_requirements) <= renewable_capacities[k])

#     for k in non_renewable_resources:
#         non_renewable_resources_requirements = [model.presence_of(ys[i][j]) * requests[k+2][i][j] for i in tasks for j in range(no_modes_list[i])]
#         model.add(model.sum(non_renewable_resources_requirements) <= non_renewable_capacities[k])

#     # define precedence constraints
#     for (i, job_successors) in enumerate(successors):
#         model.add([model.end_before_start( xs[i], xs[successor - 1] )  for successor in job_successors])

#     # solve model
#     sol = model.solve()

#     if sol.get_solve_status() in ["Unknown", "Infeasible", "JobFailed", "JobAborted"]:
#         print('No solution found')

#     if validate:
#         try:
#             print("Validating solution...")
#             validate_mmrcpsp(sol, xs, successors, sol.get_objective_value())
#             print("Solution is valid.")
#         except AssertionError as e:
#             print("Solution is invalid.")
#             print(e)
#             return None, None

#     # print solution
#     if sol.get_solve_status() == 'Optimal':
#         print("Optimal solution found")
#     elif sol.get_solve_status() == 'Feasible':
#         print("Feasible solution found")
#     else:
#         print("Unknown solution status")
#         print(sol.get_solve_status())

#     print('Objective value:', sol.get_objective_value())
#     for i in tasks:
#         print(sol.get_var_solution(xs[i]))
#         for j in range(no_modes_list[i]):
#             if sol.get_var_solution(ys[i][j]).is_absent():
#                 continue
            
#             print(f'Task {i} is scheduled in mode {j} from {sol.get_var_solution(ys[i][j]).start} to {sol.get_var_solution(ys[i][j]).end}')
#             # if sol.get_var_solution(y[i][j]) == 1:
#             #     print(f'Task {i} is scheduled in mode {j} from {sol.get_var_solution(x[i][j]).start} to {sol.get_var_solution(x[i][j]).end}')
#     # for k in resources:
#     #     print(f'Resource {k} usage: {sol.get_var_solution(z[k])}')

#     xs = [ys[i][j] for i in tasks for j in range(no_modes_list[i]) if sol.get_var_solution(ys[i][j]).is_present()]

#     return sol, xs


# def validate_mmrcpsp(sol, x, successors, horizon):
#     assert sol.get_objective_value() <= horizon, "Project completion time exceeds horizon."

#     for i, job_successors in enumerate(successors):
#         for successor in job_successors:
#             assert sol.get_var_solution(x[i]).get_end() <= sol.get_var_solution(x[successor - 1]).get_start(), f"Job {i} ends after job {successor} starts."

#     return True