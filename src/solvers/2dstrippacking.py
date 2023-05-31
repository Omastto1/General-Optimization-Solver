from docplex.cp.model import CpoModel
from collections import namedtuple

from .solver import Solver


class StripPacking2DSolver(Solver):
    def _solve_cp(self, instance, validate=False, visualize=False, force_execution=False):
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        model = CpoModel()



        print("Using 10 second time limit")
        sol = model.solve(TimeLimit=self.TimeLimit, LogVerbosity='Terse')

        if sol:
            if validate:
                try:
                    print("Validating solution...")
                    instance.validate(sol, job_operations)
                    print("Solution is valid.")
                except AssertionError as e:
                    print("Solution is invalid.")
                    print(e)
                    return None, None, None

            if visualize:
                pass
                # instance.visualize(sol, job_operations, machine_operations)

            print("Project completion time:", sol.get_objective_values()[0])
        else:
            print("No solution found.")
            
        # print solution
        if sol.get_solve_status() == 'Optimal':
            print("Optimal solution found")
        elif sol.get_solve_status() == 'Feasible':
            print("Feasible solution found")
        else:
            print("Unknown solution status")
            print(sol.get_solve_status())

        obj_value = sol.get_objective_values()[0]
        print('Objective value:', obj_value)
        instance.compare_to_reference(obj_value)

        # Solution = namedtuple("Solution", ['job_operations', 'machine_operations'])
        # variables = Solution(job_operations, machine_operations)

        instance.update_run_history(sol, variables, "CP", self.TimeLimit)

        return sol, variables
