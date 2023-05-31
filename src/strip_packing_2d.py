from src.optimization_problem import OptimizationProblem
import docplex.cp.utils_visu as visu
import matplotlib.pyplot as plt
from docplex.cp.model import CpoStepFunction


class StripPacking2D(OptimizationProblem):
    def __init__(self, benchmark_name, instance_name, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, "2DSTRIPPACKING", data, solution, run_history)

        self.no_elements = self._data["no_elements"]  # number of activities
        self.strip_width = self._data["strip_width"]
        self.widths = [element["width"] for element in self._data["elements"]]
        self.heights = [element["height"] for element in self._data["elements"]]

    def validate(self, sol, x):
        # assert sol.get_objective_value(
        # ) <= self.horizon, "Project completion time exceeds horizon."

        # for i, job_successors in enumerate(self.successors):
        #     for successor in job_successors:
        #         assert sol.get_var_solution(x[i]).get_end() <= sol.get_var_solution(
        #             x[successor - 1]).get_start(), f"Job {i} ends after job {successor} starts."
        print("WARNING: No validation implemented for 2D Strip Packing")
        return True

    def visualize(self, sol, x):
        print("WARNING: No visualization implemented for 2D Strip Packing")
        pass
        # no_jobs = len(x)

        # if sol and visu.is_visu_enabled():
        #     visu.timeline('Solution SchedOptional', 0, 110)
        #     for job_number in range(no_jobs):
        #         visu.sequence(name=job_number)
        #         wt = sol.get_var_solution(x[job_number])
        #         if wt.is_present():
        #             if wt.get_start() != wt.get_end():
        #                 visu.interval(wt, "salmon", x[job_number].get_name())
        # visu.show()

        # # Define the data for the Gantt chart
        # print(sol.get_value(x[0]))
        # start_times = [sol.get_var_solution(
        #     x[i]).get_start() for i in range(no_jobs)]
        # end_times = [sol.get_var_solution(x[i]).get_end()
        #              for i in range(no_jobs)]

        # # Create the Gantt chart
        # fig, ax = plt.subplots()
        # for i in range(no_jobs):
        #     ax.broken_barh(
        #         [(start_times[i], end_times[i] - start_times[i])], (i, 1), facecolors='blue')
        # ax.set_ylim(0, no_jobs)
        # ax.set_xlim(0, max(end_times))
        # ax.set_xlabel('Time')
        # ax.set_yticks(range(no_jobs))
        # ax.set_yticklabels(['Activity %d' % i for i in range(no_jobs)])
        # ax.grid(True)
        # plt.show()

        # # https://ibmdecisionoptimization.github.io/docplex-doc/cp/visu.rcpsp.py.html

        # if sol and visu.is_visu_enabled():
        #     load = [CpoStepFunction()
        #             for j in range(self.no_renewable_resources)]
        #     for i in range(no_jobs):
        #         itv = sol.get_var_solution(x[i])
        #         for j in range(self.no_renewable_resources):
        #             if 0 < self.requests[j][i]:
        #                 load[j].add_value(
        #                     itv.get_start(), itv.get_end(), self.requests[j][i])

        #     visu.timeline('Solution for RCPSP ')  # + filename)
        #     visu.panel('Tasks')
        #     for i in range(no_jobs):
        #         visu.interval(sol.get_var_solution(x[i]), i, x[i].get_name())
        #     for j in range(self.no_renewable_resources):
        #         visu.panel('R' + str(j+1))
        #         visu.function(
        #             segments=[(0, 200, self.renewable_capacities[j])], style='area', color='lightgrey')
        #         visu.function(segments=load[j], style='area', color=j)
        #     visu.show()
