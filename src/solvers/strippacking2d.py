from docplex.cp.model import CpoModel
from collections import namedtuple

from .solver import Solver


class StripPacking2DSolver(Solver):
    def _solve_cp(self, instance, validate=False, visualize=False, force_execution=False):
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        model = CpoModel()

        xs = [model.interval_var(name=f'element_{i}')
              for i in range(instance.no_elements)]
        ys = [[model.interval_var(size=instance.widths[i], name=f'level_{j}_element_{i}', optional=True) for i in range(
            instance.no_elements)] for j in range(
            instance.no_elements)]

        model.add(
            model.minimize(
                model.sum(
                    model.max(
                        model.presence_of(ys[level][i]) * instance.heights[i] for i in range(instance.no_elements)
                    ) for level in range(instance.no_elements)
                )
            )
        )
        
        for i in range(instance.no_elements):
            model.add(model.alternative(xs[i], [ys[level][i] for level in range(instance.no_elements)]))

        for level in ys:
            model.add(model.no_overlap(level))

        for level in range(instance.no_elements):
            model.add(model.max(model.end_of(ys[level][i]) for i in range(instance.no_elements)) <= instance.strip_width)

        print("Using 10 second time limit")
        sol = model.solve(TimeLimit=self.TimeLimit, LogVerbosity='Terse')

        # if sol:
        #     if validate:
        #         try:
        #             print("Validating solution...")
        #             instance.validate(sol, job_operations)
        #             print("Solution is valid.")
        #         except AssertionError as e:
        #             print("Solution is invalid.")
        #             print(e)
        #             return None, None, None

        #     if visualize:
        #         pass
        #         # instance.visualize(sol, job_operations, machine_operations)

        #     print("Project completion time:", sol.get_objective_values()[0])
        # else:
        #     print("No solution found.")

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

        # variables = Solution(job_operations, machine_operations)
        Solution = namedtuple("Solution", ['xs'])
        xs = [ys[level][i] for i in range(instance.no_elements) for level in range(
            instance.no_elements) if sol.get_var_solution(ys[level][i]).is_present()]
        variables = Solution(xs)

        instance.update_run_history(sol, variables, "CP", self.TimeLimit)

        return sol, variables
