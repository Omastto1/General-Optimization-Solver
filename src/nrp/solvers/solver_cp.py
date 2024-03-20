import multiprocessing
import pandas as pd

from docplex.cp.model import CpoModel
from src.common.solver import CPSolver
from src.nrp.problem import *
from tabulate import tabulate


class NRPSolver(CPSolver):
    solver_name = 'CP Integer Model'

    def build_model(self, instance):

        mdl = CpoModel(name="nurses")
        mdl.set_parameters(params=self.params)

        # first global collections to iterate upon
        all_nurses = instance.staff.keys()
        all_shifts = instance.shifts.keys()
        all_days = list(range(instance.horizon))

        # the assignment variables.
        # assigned = mdl.binary_var_cube(keys1=all_nurses, keys2=all_days, keys3=all_shifts, name="assign_%s_%s_%s")
        assigned = mdl.binary_var_dict(((n, d, s) for n in all_nurses for d in all_days for s in all_shifts), name="assign_%s_%s_%s")
        # print("assigned: ", assigned)

        # Hard Constraints

        # Employees cannot be assigned more than one shift on a day.

        for nurse in all_nurses:
            for day in all_days:
                mdl.add_constraint(mdl.sum(assigned[nurse, day, shift] for shift in all_shifts) <= 1)

        # Shift Rotation - Shifts which cannot follow the shift on the previous day are defined in SECTION_SHIFTS. This constraint always assumes that the last day of the previous planning period was a day off and the first day of the next planning horizon is a day off.

        for nurse in all_nurses:
            for day in all_days[:-1]:
                for shift in all_shifts:
                    forbidden = instance.shifts[shift]['CannotFollow']
                    # print(forbidden)
                    if len(forbidden) > 0:
                        mdl.add_constraint(
                            assigned[nurse, day, shift] + mdl.sum(assigned[nurse, day + 1, f] for f in forbidden) <= 1)

        # Maximum Number of Shifts - The maximum number of shifts of each type that can be assigned to each employee are defined in SECTION_STAFF in the field MaxShifts.

        for nurse in all_nurses:
            for shift in all_shifts:
                mdl.add_constraint(
                    mdl.sum(assigned[nurse, day, shift] for day in all_days) <= instance.staff[nurse]['max_of_shift'][shift])

        # Maximum Total Minutes - The maximum amount of total time in minutes that can be assigned to each employee is defined in SECTION_STAFF in the field MaxTotalMinutes. The duration in minutes of each shift is defined in SECTION_SHIFTS in the field Length in mins.
        # Minimum Total Minutes - The minimum amount of total time in minutes that must be assigned to each employee is defined in SECTION_STAFF in the field MinTotalMinutes. The duration in minutes of each shift is defined in SECTION_SHIFTS in the field Length in mins.

        for nurse in all_nurses:
            mdl.add_constraint(mdl.sum(
                assigned[nurse, day, shift] * instance.shifts[shift]['Length'] for day in all_days for shift in
                all_shifts) >= instance.staff[nurse]['min_total_minutes'])
            mdl.add_constraint(mdl.sum(
                assigned[nurse, day, shift] * instance.shifts[shift]['Length'] for day in all_days for shift in
                all_shifts) <= instance.staff[nurse]['max_total_minutes'])

        # Maximum Consecutive Shifts - The maximum number of consecutive shifts that can be worked before having a day off. This constraint always assumes that the last day of the previous planning period was a day off and the first day of the next planning period is a day off.

        for nurse in all_nurses:
            max_consecutive_shifts = instance.staff[nurse]['max_consecutive_shifts']
            for day in all_days[:-max_consecutive_shifts]:
                mdl.add_constraint(mdl.sum(
                    assigned[nurse, day + i, shift] for i in range(max_consecutive_shifts) for shift in
                    all_shifts) <= max_consecutive_shifts)

        # Minimum Consecutive Shifts - The minimum number of shifts that must be worked before having a day off. This constraint always assumes that there are an infinite number of consecutive shifts assigned at the end of the previous planning period and at the start of the next planning period.

        for nurse in all_nurses:
            min_consecutive_shifts = instance.staff[nurse]['min_consecutive_shifts']
            for day in all_days[:-min_consecutive_shifts]:
                mdl.add_constraint(mdl.sum(
                    assigned[nurse, day + i, shift] for i in range(min_consecutive_shifts) for shift in
                    all_shifts) >= min_consecutive_shifts)

        # Minimum Consecutive Days Off - The minimum number of consecutive days off that must be assigned before assigning a shift. This constraint always assumes that there are an infinite number of consecutive days off assigned at the end of the previous planning period and at the start of the next planning period.

        for nurse in all_nurses:
            min_consecutive_days_off = instance.staff[nurse]['min_consecutive_days_off']
            for day in all_days[:-min_consecutive_days_off]:
                mdl.add_constraint(mdl.sum(
                    assigned[nurse, day + i, shift] for i in range(min_consecutive_days_off) for shift in
                    all_shifts) == 0)

        # Maximum Number of Weekends - A weekend is defined as being worked if there is a shift on the Saturday or the Sunday.

        for nurse in all_nurses:
            max_weekends = instance.staff[nurse]['max_weekends']
            for i in range(0, len(all_days), 7):
                mdl.add_constraint(mdl.sum(
                    assigned[nurse, day, shift] for day in all_days[i:i + 2] for shift in all_shifts) <= max_weekends)

        # Days off - Shifts must not be assigned to the specified employee on the specified days. They are defined in the section SECTION_DAYS_OFF.

        for nurse in instance.days_off.keys():
            for day in instance.days_off[nurse]:
                mdl.add_constraint(mdl.sum(assigned[nurse, day, shift] for shift in all_shifts) == 0)

        # Soft Constraints
        # The following constraints are soft constraints. If they are not met then the solution's penalty is the weight value. Sum of the penalties is the total penalty for the solution and is minimized.

        penalty = []

        # Shift on requests - If the specified shift is not assigned to the specified employee on the specified day then the solution's penalty is the specified weight value. Defined in SECTION_SHIFT_ON_REQUESTS.

        for request in instance.shift_on_requests:
            nurse, day, shift, weight = request.get('EmployeeID'), request.get('Day'), request.get('ShiftID'), request.get('Weight')
            penalty.append(assigned[nurse, day, shift] * weight)

        # Shift off requests - If the specified shift is assigned to the specified employee on the specified day then the solution's penalty is the weight value. Defined in SECTION_SHIFT_OFF_REQUESTS.

        for request in instance.shift_off_requests:
            nurse, day, shift, weight = request.get('EmployeeID'), request.get('Day'), request.get('ShiftID'), request.get('Weight')
            penalty.append((1 - assigned[nurse, day, shift]) * weight)

        # Cover - If the required number of staff on the specified day for the specified shift is not assigned (defined in SECTION_COVER) then it is a soft constraint violation. If the number assigned (x) is below the required number then the solution's penalty is:
        # (requirement - x) * weight for under
        # If the total number assigned is more than the required number then the solution's penalty is:
        # (x - requirement) * weight for over

        for requirement in instance.cover_requirements:
            day, shift, requirement, weight_under, weight_over = requirement.get('Day'), requirement.get('ShiftID'), requirement.get('Requirement'), requirement.get('WeightForUnder'), requirement.get('WeightForOver')
            penalty.append((mdl.sum(assigned[nurse, day, shift] for nurse in all_nurses) - requirement) * weight_over)
            penalty.append((requirement - mdl.sum(assigned[nurse, day, shift] for nurse in all_nurses)) * weight_under)

        # Objective
        mdl.minimize(mdl.sum(penalty))

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

        if not sol or sol.get_solve_status() in ["Unknown", "Infeasible", "JobFailed", "JobAborted"]:
            print('No solution found')
            return None, None, sol

        # Fill the DataFrame with shift types
        for nurse in instance.staff.keys():
            for day in range(instance.horizon):
                for shift in instance.shifts.keys():
                    if sol.get_var_value(model_variables['assigned'][nurse, day, shift]) == 1:
                        schedule_df.at[nurse, day] = shift

        # Print the DataFrame
        # print(schedule_df.to_string())
        print(tabulate(schedule_df, headers='keys', tablefmt='psql'))

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
