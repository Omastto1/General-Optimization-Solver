import multiprocessing

from docplex.cp.model import CpoModel
from src.common.solver import CPSolver
from src.vrp.problem import *


class VRPTWSolver(CPSolver):
    solver_name = 'CP Integer Model'

    def build_model(self, instance, initial_solution=None):
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
        # pr = [(i, vrp.get_earliest_start(i), vrp.get_latest_start(i)) for i in range(vrp.get_num_visits())]
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

        if initial_solution:
            self.solver_name += " Hybrid"

            stp = model.create_empty_solution()
            # sprev = [0 for _ in range(vrp.get_num_visits())]

            # Set prev
            vehicles = list(vrp.vehicles())
            vehicle = 0
            for i, p in enumerate(initial_solution['paths']):
                # stp[prev[i]] = p[-1]
                v, fv, lv = vehicles[vehicle]
                # last = fv
                # time = 0
                # stp[start_time[fv]] = time
                # for j in p[1:-1]:
                #     j -= 1
                #     stp[veh[j]] = vehicle
                #     time = max(time + vrp.get_service_time(last) + vrp.get_distance(last, j) // TIME_FACTOR, vrp.get_earliest_start(j))
                #     stp[start_time[j]] = time
                #     last = j
                # stp[veh[lv]] = vehicle
                # stp[load[vehicle]] = sum(vrp.get_demand(j-1) for j in p)

                last = lv
                for j in p[1:-1][::-1]:
                    j -= 1
                    stp[prev[last]] = j
                    # sprev[last] = j
                    last = j
                stp[prev[last]] = fv
                # sprev[last] = fv
                stp[prev[fv]] = ((lv - 2*vrp.get_num_vehicles() - 1) % vrp.get_num_vehicles()) + 2*vrp.get_num_vehicles()
                # sprev[fv] = ((lv - 2*vrp.get_num_vehicles() - 1) % vrp.get_num_vehicles()) + 2*vrp.get_num_vehicles()

                vehicle += 1

            stp[used] = vehicle

            for v, fv, lv in vehicles[vehicle:]:
                stp[prev[fv]] = lv - 1
                stp[prev[lv]] = fv
                # sprev[fv] = lv - 1
                # sprev[lv] = fv
                stp[veh[fv]] = v
                stp[veh[lv]] = v
                stp[load[v]] = 0
                stp[start_time[fv]] = 0
                stp[start_time[lv]] = 0

            model.set_starting_point(stp)

        return model, {'vrp': vrp, 'prev': prev}

    def _export_solution(self, sol, data, model_variables):
        vrp = data
        sprev = tuple(sol.solution[p] for p in model_variables['prev'])
        print('sprev: ',sprev)

        n_vehicles = 0
        total_distance = 0
        paths = []

        for v, fv, lv in model_variables['vrp'].vehicles():
            route = []
            nd = lv
            while nd != fv:
                route.append(nd)
                nd = sprev[nd]
            route.append(fv)
            route.reverse()

            if len(route) > 2:
                n_vehicles += 1
                paths.append([0 if i + 1 > vrp.nb_customers else i + 1 for i in route])

                for idx, nd in enumerate(route):
                    if nd != route[-1]:
                        nxt = route[idx + 1]
                        locald = model_variables['vrp'].get_distance(nd, nxt)
                        total_distance += locald

        total_distance /= TIME_FACTOR
        ret = {'n_vehicles': n_vehicles, 'total_distance': total_distance, 'paths': paths}
        return ret

    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True, initial_solution=None):
        print("Building model")
        model, model_variables = self.build_model(instance, initial_solution=initial_solution)

        print("Looking for solution")
        sol = model.solve()

        if sol.get_solve_status() in ["Unknown", "Infeasible", "JobFailed", "JobAborted"]:
            print('No solution found')
            return None, None, sol

        result = self._export_solution(sol, instance, model_variables)

        if validate:
            try:
                print("Validating solution...")
                is_valid = validate_path(result, instance)
                if is_valid:
                    print("Solution is valid.")
                else:
                    print("Solution is invalid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, None, result

        if visualize:
            visualize_path(result, instance)

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

        return obj_value, result, sol
