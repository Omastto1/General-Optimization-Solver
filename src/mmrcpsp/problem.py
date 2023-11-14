from src.rcpsp.problem import RCPSP
import docplex.cp.utils_visu as visu
from docplex.cp.model import CpoStepFunction


class MMRCPSP(RCPSP):
    def __init__(self, benchmark_name, instance_name, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, data, solution, run_history)
        self._instance_kind = "MMRCPSP"

        # new
        # number of non-renewable resources
        self.no_non_renewable_resources = self._data["resources"][
            "non_renewable_resources"]["number_of_resources"]
        # available resource capacity
        self.non_renewable_capacities = self._data["resources"][
            "non_renewable_resources"]["non_renewable_availabilities"]
        self.no_modes_list = [job_specification['no_modes']
                              for job_specification in self._data['job_specifications']]

        resource_keys = ["R1", "R2", "N1", "N2"]
        # overwritte rcpsp variables
        self.durations = [[mode["duration"] for mode in job["modes"]]
                          for job in self._data["job_specifications"]]  # duration of each activity
        self.requests = [[[mode["request_duration"][k] for mode in job["modes"]]
                          for job in self._data["job_specifications"]] for k in resource_keys]

    def validate(self, export):  # sol, x, 
        super().validate(None, None, export)  # sol, x, 

    def visualize(self, export):  # , sol, xs
        # https://ibmdecisionoptimization.github.io/docplex-doc/cp/visu.rcpsp.py.html

        # if sol and visu.is_visu_enabled():
        #     if xs is not None:
        #         load = [CpoStepFunction() for j in range(
        #             self.no_renewable_resources + self.no_non_renewable_resources)]
        #         for i, _x in enumerate(xs[1:-1], start=1):

        #             itv = sol.get_var_solution(_x)
        #             print("itv solution", itv.get_start(), itv.get_end())
        #             print(_x.get_name())
        #             mode = int(_x.get_name().split("_")[-1])
        #             print("mode", mode)
        #             print("ranges", self.no_renewable_resources, self.no_jobs)
        #             for j in range(self.no_renewable_resources):
        #                 print(j, i, mode)
        #                 if 0 < self.requests[j][i][mode-1]:
        #                     load[j].add_value(
        #                         itv.get_start(), itv.get_end(), self.requests[j][i][mode-1])
        #             for j in range(2, self.no_non_renewable_resources + 2):
        #                 if 0 < self.requests[j][i][mode-1]:
        #                     load[j].add_value(
        #                         itv.get_start(), sol.get_objective_value(), self.requests[j][i][mode-1])

        #         visu.timeline('Solution for MMRCPSP ')  # + filename)
        #         visu.panel('Tasks')
        #         for _x in xs[1:-1]:
        #             visu.interval(sol.get_var_solution(_x), i, "_".join(_x.get_name().split("_")[:2] + _x.get_name().split("_")[3:]))

        #         for j in range(self.no_renewable_resources):
        #             visu.panel('R' + str(j+1))
        #             visu.function(segments=[
        #                         (0, 200, self.renewable_capacities[j])], style='area', color='lightgrey')
        #             visu.function(segments=load[j], style='area', color=j)

        #         for j in range(2, self.no_non_renewable_resources + 2):
        #             visu.panel('NR' + str(j+1))
        #             visu.function(segments=[
        #                         (0, 200, self.non_renewable_capacities[j-2])], style='area', color='lightgrey')
        #             visu.function(segments=load[j], style='area', color=j)

        #         visu.show()
        # else:
        obj_value = max([x['end'] for x in export])
        load = [CpoStepFunction() for j in range(
            self.no_renewable_resources + self.no_non_renewable_resources)]
        for i, _x in enumerate(export[1:-1], start=1):
            print("itv solution", _x['start'], _x['end'])
            print(_x['name'])
            mode = int(_x['name'].split("_")[-1])
            print("mode", mode)
            print("ranges", self.no_renewable_resources, self.no_jobs)
            for j in range(self.no_renewable_resources):
                print(j, i, mode)
                if 0 < self.requests[j][i][mode]:
                    load[j].add_value(
                        _x['start'], _x['end'], self.requests[j][i][mode])
            for j in range(2, self.no_non_renewable_resources + 2):
                if 0 < self.requests[j][i][mode]:
                    load[j].add_value(
                        _x['start'], obj_value, self.requests[j][i][mode])

        visu.timeline('Solution for MMRCPSP ')  # + filename)
        visu.panel('Tasks') 
        for _x in export[1:-1]:
            # visu.interval(sol.get_var_solution(_x), i, "_".join(_x.get_name().split("_")[:2] + _x.get_name().split("_")[3:]))
            visu.interval(_x['start'], _x['end'], i, "_".join(_x['name'].split("_")[:2] + _x['name'].split("_")[3:]))

        for j in range(self.no_renewable_resources):
            visu.panel('R' + str(j+1))
            visu.function(segments=[
                        (0, 200, self.renewable_capacities[j])], style='area', color='lightgrey')
            visu.function(segments=load[j], style='area', color=j)

        for j in range(2, self.no_non_renewable_resources + 2):
            visu.panel('NR' + str(j+1))
            visu.function(segments=[
                        (0, 200, self.non_renewable_capacities[j-2])], style='area', color='lightgrey')
            visu.function(segments=load[j], style='area', color=j)

        visu.show()

