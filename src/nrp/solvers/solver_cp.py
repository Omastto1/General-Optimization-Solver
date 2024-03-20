import multiprocessing
import pandas as pd

from docplex.mp.model import Model
from src.common.solver import CPSolver
from src.nrp.problem import *
from tabulate import tabulate


class NRPSolver(CPSolver):
    solver_name = 'CP Integer Model'

    def build_model(self, instance):

        mdl = Model(name="nurses")

        # first global collections to iterate upon
        all_nurses = instance.staff.keys()
        all_shifts = instance.shifts.keys()
        all_days = list(range(instance.horizon))

        # the assignment variables.
        assigned = mdl.binary_var_cube(keys1=all_nurses, keys2=all_days, keys3=all_shifts, name="assign_%s_%s_%s")
        # print("assigned: ", assigned)
        # vacations

        vacations = []

        for nurse in instance.days_off.keys():
            for day in instance.days_off[nurse]:
                # print(nurse, day)
                for shift in instance.cover_requirements:
                    if shift['Day'] == day:
                        temp = assigned[nurse, shift['Day'], shift['ShiftID']]
                        vacations.append(temp)
                        mdl.add_constraint(temp == 0)
                # for day in days:
                #     vacations.append(assigned[nurse, day])

        # print("vacations: ", vacations)

        # max 1 shift per day

        for nurse in all_nurses:
            for day in all_days:
                mdl.add_constraint(mdl.sum(assigned[nurse, day, shift] for shift in all_shifts) <= 1)

        # eliminate following shifts

        for nurse in all_nurses:
            for day in all_days[:-1]:
                for shift in all_shifts:
                    forbidden = instance.shifts[shift]['CannotFollow']
                    # print(forbidden)
                    if len(forbidden) > 0:
                        mdl.add_constraint(
                            assigned[nurse, day, shift] + mdl.sum(assigned[nurse, day + 1, f] for f in forbidden) <= 1)

        # The maximum numbers of shifts of each type that can be assigned to employees.

        for nurse in all_nurses:
            for shift in all_shifts:
                mdl.add_constraint(
                    mdl.sum(assigned[nurse, day, shift] for day in all_days) <= instance.staff[nurse]['max_of_shift'][shift])

        # Minimum and maximum work time

        for nurse in all_nurses:
            mdl.add_constraint(mdl.sum(
                assigned[nurse, day, shift] * instance.shifts[shift]['Length'] for day in all_days for shift in
                all_shifts) >= instance.staff[nurse]['min_total_minutes'])
            mdl.add_constraint(mdl.sum(
                assigned[nurse, day, shift] * instance.shifts[shift]['Length'] for day in all_days for shift in
                all_shifts) <= instance.staff[nurse]['max_total_minutes'])

        # Maximum consecutive shifts. The maximum number of shifts an employee can work without a day off.
        # Each nurse can work at most 'max_consecutive_shifts' consecutive days.

        for nurse in all_nurses:
            max_consecutive_shifts = instance.staff[nurse]['max_consecutive_shifts']
            for day in all_days[:-max_consecutive_shifts]:
                mdl.add_constraint(mdl.sum(
                    assigned[nurse, day + i, shift] for i in range(max_consecutive_shifts) for shift in
                    all_shifts) <= max_consecutive_shifts)

        # Minimum consecutive shifts.

        # for nurse in all_nurses:
        #     min_consecutive_shifts = instance.staff[nurse]['min_consecutive_shifts']
        #     for day in all_days[:-min_consecutive_shifts]:
        #         mdl.add_constraint(mdl.sum(
        #             assigned[nurse, day + i, shift] for i in range(min_consecutive_shifts) for shift in
        #             all_shifts) >= min_consecutive_shifts)

        # Minimum consecutive days off

        # for nurse in all_nurses:
        #     min_consecutive_days_off = instance.staff[nurse]['min_consecutive_days_off']
        #     for day in all_days[:-min_consecutive_days_off]:
        #         mdl.add_constraint(mdl.sum(
        #             assigned[nurse, day + i, shift] for i in range(min_consecutive_days_off) for shift in
        #             all_shifts) == 0)

        return mdl, {"assigned": assigned}

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
