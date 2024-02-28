from docplex.cp.model import CpoModel
from src.common.solver import CPSolver
from src.vrp.problem import *


class VRPTWSolver(CPSolver):
    def build_model(self, instance):
        vrp = VRP(instance)

        model = CpoModel()
        model.set_parameters(params=self.params)

        # Prev variables, circuit, first/last
        prev = [model.integer_var(0, vrp.get_num_visits() - 1, "P{}".format(i)) for i in range(vrp.get_num_visits())]
        for v, fv, lv in vrp.vehicles():
            model.add(prev[fv] == vrp.get_last((v - 1) % vrp.get_num_vehicles()))

        before = vrp.customers() + vrp.first()
        for c in vrp.customers():
            model.add(model.allowed_assignments(prev[c], before))
            model.add(prev[c] != c)

        for _, fv, lv in vrp.vehicles():
            model.add(model.allowed_assignments(prev[lv], vrp.customers() + (fv,)))

        model.add(model.sub_circuit(prev))

        # Vehicle
        veh = [model.integer_var(0, vrp.get_num_vehicles() - 1, "V{}".format(i)) for i in range(vrp.get_num_visits())]
        for v, fv, lv in vrp.vehicles():
            model.add(veh[fv] == v)
            model.add(veh[lv] == v)
            model.add(model.element(veh, prev[lv]) == v)
        for c in vrp.customers():
            model.add(veh[c] == model.element(veh, prev[c]))

        # Demand
        load = [model.integer_var(0, vrp.get_capacity(), "L{}".format(i)) for i in range(vrp.get_num_vehicles())]
        used = model.integer_var(0, vrp.get_num_vehicles(), 'U')
        cust_veh = [veh[c] for c in vrp.customers()]
        demand = [vrp.get_demand(c) for c in vrp.customers()]
        model.add(model.pack(load, cust_veh, demand, used))

        # Time
        start_time = [model.integer_var(vrp.get_earliest_start(i), vrp.get_latest_start(i), "T{}".format(i)) for i in
                      range(vrp.get_num_visits())]
        for fv in vrp.first():
            model.add(start_time[fv] == 0)
        for i in vrp.customers() + vrp.last():
            arrive = model.element(
                [start_time[j] + vrp.get_service_time(j) + vrp.get_distance(j, i) for j in range(vrp.get_num_visits())],
                prev[i])
            model.add(start_time[i] == model.max(arrive, vrp.get_earliest_start(i)))

        # Distance
        all_dist = []
        for i in vrp.customers() + vrp.last():
            ldist = [vrp.get_distance(j, i) for j in range(vrp.get_num_visits())]
            all_dist.append(model.element(ldist, prev[i]))
        total_distance = model.sum(all_dist) / TIME_FACTOR

        # Variables with inferred values
        model.add(model.inferred(cust_veh + load + [used] + start_time))

        # Objective
        model.add(model.minimize(total_distance))

        # KPIs
        model.add_kpi(used, 'Used')

        return model, {'vehicles': veh}

    def _export_solution(self, instance, sol, model_variables):
        # TODO: sol to path
        pass

    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True):
        print("Building model")
        model, model_variables = self.build_model(instance)

        print("Looking for solution")
        sol = model.solve()

        if sol.get_solve_status() in ["Unknown", "Infeasible", "JobFailed", "JobAborted"]:
            print('No solution found')
            return None, None, sol

        # model_variables_export = self._export_solution(instance, sol, model_variables)

        if validate:
            try:
                print("Validating solution...")
                is_valid = validate_path(sol, instance)
                if is_valid:
                    print("Solution is valid.")
                else:
                    print("Solution is invalid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, None, sol

        if visualize:
            instance.visualize_path(sol, instance)

        obj_value = sol.get_objective_values()[0]
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

        return obj_value, model_variables, sol
