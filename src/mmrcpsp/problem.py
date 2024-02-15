from src.rcpsp.problem import RCPSP
import docplex.cp.utils_visu as visu
from docplex.cp.model import CpoStepFunction


class MMRCPSP(RCPSP):
    def __init__(self, benchmark_name, instance_name, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, data, solution, run_history)
        self._instance_kind = "MMRCPSP"

        self.no_non_renewable_resources = self._data["resources"][
            "non_renewable_resources"]["number_of_resources"]
        self.non_renewable_capacities = self._data["resources"][
            "non_renewable_resources"]["non_renewable_availabilities"]
        self.no_modes_list = [job_specification['no_modes']
                              for job_specification in self._data['job_specifications']]

        resource_keys = ["R1", "R2", "N1", "N2"]

        self.durations = [[mode["duration"] for mode in job["modes"]]
                          for job in self._data["job_specifications"]]  # duration of each activity
        self.requests = [[[mode["request_duration"][k] for mode in job["modes"]]
                          for job in self._data["job_specifications"]] for k in resource_keys]

    def validate(self, model_variables_export):  # sol, x,
        model_variables_export = {
            "tasks_schedule": model_variables_export['task_mode_assignment']}
        return super().validate(None, None, model_variables_export)  # sol, x,

    def visualize(self, model_variables_export):  # , sol, xs
        # https://ibmdecisionoptimization.github.io/docplex-doc/cp/visu.rcpsp.py.html

        export = model_variables_export['task_mode_assignment']

        obj_value = max([x['end'] for x in export])
        load = [CpoStepFunction() for j in range(
            self.no_renewable_resources + self.no_non_renewable_resources)]
        for i, _x in enumerate(export[1:-1], start=1):
            mode = int(_x['name'].split("_")[-1])
            for j in range(self.no_renewable_resources):
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
            visu.interval(_x['start'], _x['end'], i, "_".join(
                _x['name'].split("_")[:2] + _x['name'].split("_")[3:]))

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
