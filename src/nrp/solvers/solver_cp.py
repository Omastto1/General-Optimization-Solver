import numpy as np
import pandas as pd
from docplex.cp.model import CpoModel

from src.common.solver import CPSolver
from src.nrp.problem import *


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
        assigned = mdl.binary_var_dict(((n, d, s) for n in all_nurses for d in all_days for s in all_shifts),
                                       name="assign_%s_%s_%s")
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
                    mdl.sum(assigned[nurse, day, shift] for day in all_days) <= instance.staff[nurse]['max_of_shift'][
                        shift])

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
                    assigned[nurse, day + i, shift] for i in range(max_consecutive_shifts+1) for shift in all_shifts)
                                   <= max_consecutive_shifts)

        # Minimum Consecutive Shifts - The minimum number of shifts that must be worked before having a day off. This constraint always assumes that there are an infinite number of consecutive shifts assigned at the end of the previous planning period and at the start of the next planning period.

        for nurse in all_nurses:
            for s in range(1, instance.staff[nurse]['min_consecutive_shifts']):  # min_consecutive_shifts = 2
                for day in all_days[:-s - 1]:
                    mdl.add_constraint(mdl.sum(assigned[nurse, day, shift] for shift in all_shifts) +  # 0
                                       s - mdl.sum(
                        assigned[nurse, day + i, shift] for i in range(1, s + 1) for shift in all_shifts) +     # + 3 - 0
                                       mdl.sum(assigned[nurse, day + s + 1, shift] for shift in all_shifts) > 0)  # 0

        # Minimum Consecutive Days Off - The minimum number of consecutive days off that must be assigned before assigning a shift. This constraint always assumes that there are an infinite number of consecutive days off assigned at the end of the previous planning period and at the start of the next planning period.

        for nurse in all_nurses:
            for s in range(1, instance.staff[nurse]['min_consecutive_days_off'] - 1):
                for day in all_days[:-s + 1]:
                    mdl.add_constraint(1 - mdl.sum(assigned[nurse, day, shift] for shift in all_shifts) +
                                       mdl.sum(assigned[nurse, day + i, shift] for i in range(1, s) for shift in
                                               all_shifts) +
                                       1 - mdl.sum(assigned[nurse, day + s + 1, shift] for shift in all_shifts) > 0)

        # Maximum Number of Weekends - A weekend is defined as being worked if there is a shift on the Saturday or the Sunday.
        # Variable k represents if nurse n has a shift on weekend w. If k=1 then the nurse has a shift on the weekend, if k=0 then the nurse does not have a shift on the weekend.

        number_of_weeks = instance.horizon // 7
        k = mdl.binary_var_dict(((n, w) for n in all_nurses for w in range(number_of_weeks)), name="weekend_%s_%s")

        for nurse in all_nurses:
            for w in range(number_of_weeks):
                saturday = w * 7 + 5

                mdl.add_constraint(
                    mdl.sum(assigned[nurse, d, shift] for d in range(saturday, saturday + 2) for shift in all_shifts) >=
                    k[nurse, w])
                mdl.add_constraint(
                    mdl.sum(assigned[nurse, d, shift] for d in range(saturday, saturday + 2) for shift in all_shifts) <=
                    k[nurse, w])

        for nurse in all_nurses:
            mdl.add_constraint(
                mdl.sum(k[nurse, w] for w in range(number_of_weeks)) <= instance.staff[nurse]['max_weekends'])

        # Days off - Shifts must not be assigned to the specified employee on the specified days. They are defined in the section SECTION_DAYS_OFF.

        for nurse in instance.days_off.keys():
            for day in instance.days_off[nurse]:
                for shift in all_shifts:
                    mdl.add_constraint(assigned[nurse, day, shift] == 0)

        # Soft Constraints
        # The following constraints are soft constraints. If they are not met then the solution's penalty is the weight value. Sum of the penalties is the total penalty for the solution and is minimized.

        penalty = []

        # Shift on requests - If the specified shift is not assigned to the specified employee on the specified day then the solution's penalty is the specified weight value. Defined in SECTION_SHIFT_ON_REQUESTS.

        for request in instance.shift_on_requests:
            nurse, day, shift, weight = request.get('EmployeeID'), request.get('Day'), request.get(
                'ShiftID'), request.get('Weight')
            penalty.append((1 - assigned[nurse, day, shift]) * weight)

        # Shift off requests - If the specified shift is assigned to the specified employee on the specified day then the solution's penalty is the weight value. Defined in SECTION_SHIFT_OFF_REQUESTS.

        for request in instance.shift_off_requests:
            nurse, day, shift, weight = request.get('EmployeeID'), request.get('Day'), request.get(
                'ShiftID'), request.get('Weight')
            penalty.append(assigned[nurse, day, shift] * weight)

        # Cover - If the required number of staff on the specified day for the specified shift is not assigned (defined in SECTION_COVER) then it is a soft constraint violation. If the number assigned (x) is below the required number then the solution's penalty is:
        # (requirement - x) * weight for under
        # If the total number assigned is more than the required number then the solution's penalty is:
        # (x - requirement) * weight for over
        # Variable z represents total above the preferred cover for shift type t on day d.
        z = mdl.integer_var_dict(((d, s) for d in all_days for s in all_shifts), name="above_%s_%s", min=0)
        # Variable y represents total below the preferred cover for shift type t on day d.
        y = mdl.integer_var_dict(((d, s) for d in all_days for s in all_shifts), name="below_%s_%s", min=0)

        for cover in instance.cover_requirements:
            day, shift, preferred, weight_under, weight_over = cover.get('Day'), cover.get('ShiftID'), cover.get(
                'Requirement'), cover.get('WeightForUnder'), cover.get('WeightForOver')
            mdl.add_constraint(mdl.sum(assigned[nurse, day, shift] for nurse in all_nurses) - z[day, shift] + y[
                day, shift] == preferred)
            penalty.append(y[day, shift] * weight_under)
            penalty.append(z[day, shift] * weight_over)

        # Objective
        mdl.minimize(mdl.sum(penalty))

        return mdl, {"assigned": assigned}

    def _export_solution(self, sol, instance, model_variables):
        result = np.full((len(instance.staff), instance.horizon), " ", dtype=str)

        staff_keys = list(instance.staff.keys())

        for nurse in instance.staff.keys():
            for day in range(instance.horizon):
                for shift in instance.shifts.keys():
                    if sol.get_value(model_variables['assigned'][nurse, day, shift]) == 1:
                        nurse_int = staff_keys.index(nurse)
                        assert result[nurse_int, day] == " ", \
                            f"Shift already assigned to nurse {nurse} on day {day}, old shift: {result[nurse_int, day]}, new shift: {shift}"
                        result[nurse_int, day] = shift

        return result

    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True):
        print("Building model")
        model, model_variables = self.build_model(instance)

        print("Looking for solution")
        sol = model.solve()

        if not sol or sol.get_solve_status() in ["Unknown", "Infeasible", "JobFailed", "JobAborted"]:
            print('No solution found')
            return None, None, sol

        result = self._export_solution(sol, instance, model_variables)
        obj_value = sol.get_objective_values()[0]

        if visualize:
            visualize_nrp(result, instance, obj_value)

        if validate:
            try:
                print("Validating solution...")
                is_valid = validate_nrp(instance, sol, result)
                if is_valid:
                    print("Solution is valid.")
                else:
                    print("Solution is invalid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, None, result

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

            instance._run_history[-1]["solution_info"]['roster'] = result

        return obj_value, model_variables, sol
