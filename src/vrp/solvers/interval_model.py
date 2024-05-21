import numpy as np

from src.common.solver import CPSolver
from src.vrp.problem import *

import docplex
from docplex.cp.model import *
from docplex.cp.model import CpoModel


class VRPTWSolver(CPSolver):
    solver_name = 'CP Interval Model'

    def build_model(self, instance):
        vrp = VRP(instance)
        num_cust = vrp.get_num_customers()
        num_vehicles = vrp.get_num_vehicles()
        n = vrp.get_num_visits()

        # print("num_cust=", num_cust)
        # print("num_vehicles=", num_vehicles)
        # print("n=", n)

        mdl = CpoModel()
        mdl.set_parameters(params=self.params)

        visit = [mdl.interval_var(name="V{}".format(i)) for i in range(n)]

        visit_veh = [[mdl.interval_var(optional=True, size=vrp.get_service_time(i),
                                       start=(vrp.get_earliest_start(i), INTERVAL_MAX),
                                       end=(0, vrp.get_latest_start(i) + vrp.get_service_time(i)),
                                       name="V{}_{}".format(i, vehicle))
                      if i != num_cust + vehicle and i != num_cust + num_vehicles + vehicle else
                      mdl.interval_var(size=vrp.get_service_time(i), start=(vrp.get_earliest_start(i), INTERVAL_MAX),
                                       end=(0, vrp.get_latest_start(i) + vrp.get_service_time(i)),
                                       name="V{}_{}".format(i, vehicle))
                      for i in range(n)] for vehicle in range(num_vehicles)]

        route = [mdl.sequence_var([visit_veh[vehicle][i] for i in range(n)], types=[i for i in range(n)],
                                  name="R{}".format(vehicle)) for vehicle in range(num_vehicles)]

        all_dist = []
        for vehicle in range(num_vehicles):
            for curr in range(n):
                all_dist.append(mdl.element(vrp.distance_matrix[curr],
                                            mdl.type_of_next(route[vehicle], visit_veh[vehicle][curr], curr, curr)))
        total_distance = mdl.sum(all_dist) / TIME_FACTOR

        load = [mdl.sum([mdl.presence_of(visit_veh[vehicle][i]) * vrp.get_demand(i) for i in range(num_cust)]) for
                vehicle in range(num_vehicles)]

        mdl.add(mdl.minimize(total_distance))

        for vehicle in range(num_vehicles):
            mdl.add(mdl.no_overlap(route[vehicle], vrp.distance_matrix))
            mdl.add(mdl.start_of(visit_veh[vehicle][num_cust + vehicle]) == 0)
            mdl.add(mdl.first(route[vehicle], visit_veh[vehicle][num_cust + vehicle]))
            mdl.add(mdl.last(route[vehicle], visit_veh[vehicle][num_cust + num_vehicles + vehicle]))

            mdl.add(load[vehicle] <= vrp.get_capacity())

        for i in range(n):
            mdl.add(mdl.alternative(visit[i], [visit_veh[k][i] for k in range(num_vehicles)]))

        return mdl, {"visit": visit, "visit_veh": visit_veh, "route": route}

    def _export_solution(self, sol, data, model_variables):
        vrp = data

        n_vehicles = 0
        total_distance = np.round(sol.solution.objective_values[0], 3)
        paths = []

        solutions = []

        for key, sol in sol.solution.var_solutions_dict.items():
            match type(sol):

                case docplex.cp.solution.CpoSequenceVarSolution:
                    if sol.lvars != []:
                        # print(sol)
                        if len(sol.lvars) > 2:
                            solutions.append(sol)
                            # print("found path", key)

        # convert sequence to path
        paths = []
        for sol in solutions:
            path = []
            for i in range(len(sol.lvars)):
                target = int(sol.lvars[i].get_name()[1:].split('_')[0]) + 1
                if target > vrp.nb_customers:
                    target = 0
                path.append(target)
            if path in paths:
                continue
            # path.reverse()
            paths.append(path)

        ret = {'n_vehicles': len(paths), 'total_distance': total_distance, 'paths': paths}
        return ret

    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True, initial_solution=None):
        print("Building model")
        model, model_variables = self.build_model(instance)

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

        return obj_value, model_variables, sol
