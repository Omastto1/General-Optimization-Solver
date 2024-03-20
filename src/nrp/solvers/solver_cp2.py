import multiprocessing
import pandas as pd

from docplex.cp.model import CpoModel
from src.common.solver import CPSolver
from src.nrp.problem import *
from tabulate import tabulate


class NRPSolver(CPSolver):
    solver_name = 'CP Integer Model'

    def build_model(self, instance):

        model = CpoModel(name="nurses")

        # All shifts including a 'VACATION' shift
        all_shifts = list(instance.shifts.keys()) + ['VACATION']

        # Define sequence variables for each nurse
        assigned = [model.sequence_var([shift for shift in range(len(all_shifts))], types=[shift for shift in range(len(all_shifts))], name=f"nurse_{nurse}") for nurse in
                    instance.staff.keys()]
        # assigned = [model.sequence_var([shift for shift in all_shifts], types=[shift for shift in all_shifts],
        #                              name="nurse_{}".format(nurse)) for nurse in all_nurses]

        # Hard Constraints

        # 1. Employees cannot be assigned more than one shift on a day
        for sequence in assigned:
            model.add(model.no_overlap(sequence))

        # 2. Shift Rotation
        # Implementing this constraint depends on the specific shifts that cannot follow each other.

        # 3. Maximum Number of Shifts
        for nurse, sequence in zip(instance.staff.keys(), assigned):
            max_shifts = instance.staff[nurse]["MaxShifts"]
            model.add(model.sum(1 for shift in all_shifts for i in range(instance.horizon) if
                                model.always_in(shift, sequence, i)) <= max_shifts)

        # 4. Maximum Total Minutes
        for nurse, sequence in zip(instance.staff.keys(), assigned):
            max_total_minutes = instance.staff[nurse]["MaxTotalMinutes"]
            duration_map = {shift: instance.shifts[shift]["Length in mins"] for shift in instance.shifts.keys()}
            model.add(model.sum(duration_map[shift] for shift in all_shifts for i in range(instance.horizon) if
                                model.always_in(shift, sequence, i)) <= max_total_minutes)

        # 5. Minimum Total Minutes
        for nurse, sequence in zip(instance.staff.keys(), assigned):
            min_total_minutes = instance.staff[nurse]["MinTotalMinutes"]
            duration_map = {shift: instance.shifts[shift]["Length in mins"] for shift in instance.shifts.keys()}
            model.add(model.sum(duration_map[shift] for shift in all_shifts for i in range(instance.horizon) if
                                model.always_in(shift, sequence, i)) >= min_total_minutes)

        # 6. Maximum Consecutive Shifts
        # Implementing this constraint depends on the specific maximum consecutive shifts allowed.

        # 7. Minimum Consecutive Shifts
        # Implementing this constraint depends on the specific minimum consecutive shifts required.

        # 8. Minimum Consecutive Days Off
        # Implementing this constraint depends on the specific minimum consecutive days off required.

        # 9. Maximum Number of Weekends
        # Implementing this constraint depends on whether there are shifts on Saturdays and Sundays.

        # 10. Days off
        # Implementing this constraint depends on the specified days off for each nurse.

        # Soft Constraints

        # 1. Shift on requests
        # Implementing this constraint depends on the specified shift on requests.

        # 2. Shift off requests
        # Implementing this constraint depends on the specified shift off requests.

        # 3. Cover
        # Implementing this constraint depends on the required number of staff for each shift.

        return model, {"assigned": assigned}

    def _export_solution(self, sol, data, model_variables):
        pass

    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True):
        print("Building model")
        model, model_variables = self.build_model(instance)

        print("Looking for solution")
        sol = model.solve()

        # Print the result in a nice table

        schedule_df = pd.DataFrame(0, index=instance.staff.keys(), columns=range(instance.horizon))

        # Fill the DataFrame with shift types
        for nurse in instance.staff.keys():
            for day in range(instance.horizon):
                for shift in instance.shifts.keys():
                    if sol.get_var_value(model_variables['assigned'][nurse, day, shift]) == 1:
                        schedule_df.at[nurse, day] = shift

        # Print the DataFrame
        # print(schedule_df.to_string())
        print(tabulate(schedule_df, headers='keys', tablefmt='psql'))

        if sol.get_solve_status() in ["Unknown", "Infeasible", "JobFailed", "JobAborted"]:
            print('No solution found')
            return None, None, sol

        result = self._export_solution(sol, instance, model_variables)

        if validate:
            try:
                print("Validating solution...")
                is_valid = validate_nrp(result, instance)
                if is_valid:
                    print("Solution is valid.")
                else:
                    print("Solution is invalid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, None, result

        if visualize:
            visualize_nrp(result, instance)

        obj_value = result['total_distance']
        print('Objective value:', obj_value)

        if sol.get_solve_status() == 'Optimal':
            print("Optimal solution found")
        elif sol.get_solve_status() == 'Feasible':
            print("Feasible solution found")
        else:
            print("Unknown solution status")
            print(sol.get_solve_status())

        instance.compare_to_reference(obj_value)

        if update_history:
            self.add_run_to_history(instance, sol)

            # Add number of vehicles and total distance and paths to history
            instance._run_history[-1]["solution_info"]['n_vehicles'] = result['n_vehicles']
            instance._run_history[-1]["solution_info"]['total_distance'] = result['total_distance']
            instance._run_history[-1]["solution_info"]['paths'] = result['paths']

        return obj_value, model_variables, sol
