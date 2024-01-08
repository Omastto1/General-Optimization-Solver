from docplex.cp.model import CpoModel
from collections import namedtuple

from src.common.solver import CPSolver


class StripPackingLeveled2DCPSolver(CPSolver):
    solver_name = 'CP Default Leveled'

    def _export_solution(self, instance, solution, xs, ys):
        variables_export = [None] * instance.no_elements

        height = 0
        for level in range(instance.no_elements):
            level_variable_solutions = [solution.get_var_solution(ys[level][item]) for item in range(instance.no_elements)]

            if not any(variable_solution.is_present() for variable_solution in level_variable_solutions):
                continue

            level_height = max(instance.rectangles[item]['height'] for item, variable_solution in enumerate(level_variable_solutions) if variable_solution.is_present())

            for item in range(instance.no_elements):
                if level_variable_solutions[item].is_present():
                    variables_export[item] = (solution.get_var_solution(ys[level][item]).get_start(), height)

            height += level_height

        return variables_export
    
    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True):
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        model = CpoModel()
        model.set_parameters(params=self.params)

        xs = [model.interval_var(name=f'element_{i}')
              for i in range(instance.no_elements)]
        ys = [[model.interval_var(size=instance.rectangles[i]['width'], name=f'level_{j}_element_{i}', optional=True) for i in range(
            instance.no_elements)] for j in range(
            instance.no_elements)]

        model.add(
            model.minimize(
                model.sum(
                    model.max(
                        model.presence_of(ys[level][i]) * instance.rectangles[i]['height'] for i in range(instance.no_elements)
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
        sol = model.solve()  # TimeLimit=self.TimeLimit, LogVerbosity='Terse'

        if sol.get_solve_status() in ["Unknown", "Infeasible", "JobFailed", "JobAborted"]:
            print('No solution found')
            return None, None, sol
        
        
        placements = self._export_solution(instance, sol, xs, ys)
        total_height = sol.get_objective_values()[0]
        
        if validate:
            try:
                print("Validating solution...")
                is_valid = instance.validate(sol)
                if is_valid:
                    print("Solution is valid.")
                else:
                    print("Solution is invalid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, None, None

        if visualize:
            instance.visualize(sol, placements, total_height)

        print('Objective value:', total_height)

        if sol.get_solve_status() == 'Optimal':
            print("Optimal solution found")
        elif sol.get_solve_status() == 'Feasible':
            print("Feasible solution found")
        else:
            print("Unknown solution status")
            print(sol.get_solve_status())

        print(sol.solution.get_objective_bounds())
        print(sol.solution.get_objective_gaps())
        print(sol.solution.get_objective_values())

        instance.compare_to_reference(total_height)

        if update_history:
            self.add_run_to_history(instance, sol)

        return total_height, placements, sol
