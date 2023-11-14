from src.common.optimization_problem import OptimizationProblem
import docplex.cp.utils_visu as visu
import matplotlib.pyplot as plt
from docplex.cp.model import CpoStepFunction
from docplex.cp.solution import CpoIntervalVarSolution


class RCPSP(OptimizationProblem):
    def __init__(self, benchmark_name, instance_name, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, "RCPSP", data, solution, run_history)

        self.no_jobs = self._data["number_of_jobs"]  # number of activities
        self.no_renewable_resources = self._data["resources"]["renewable_resources"]["number_of_resources"]
        self.horizon = self._data["horizon"] if "horizon" in self._data else 2 ** 32
        self.durations = [job["modes"][0]["duration"]
                          for job in self._data["job_specifications"]]  # duration of each activity
        # precedence constraints
        self.successors = [job["successors"]
                           for job in self._data["job_specifications"]]
        # available resource capacity
        self.renewable_capacities = self._data["resources"]["renewable_resources"]["renewable_availabilities"]
        self.requests = [[self._data["job_specifications"][i]["modes"][0]["request_duration"]
                          [f"R{k+1}"] for i in range(self.no_jobs)] for k in range(self.no_renewable_resources)]

    def validate(self, sol, x, xs=None):  #
        """ xs holding arrays of "start", "end" dictionaries """
        # TODO:
        if sol is not None:
            assert sol.get_objective_value(
            ) <= self.horizon, "Project completion time exceeds horizon."

            for i, job_successors in enumerate(self.successors):
                for successor in job_successors:
                    assert sol.get_var_solution(x[i]).get_end() <= sol.get_var_solution(
                        x[successor - 1]).get_start(), f"Job {i} ends after job {successor} starts."
            
            assert sol.get_var_solution(x[0]).get_start() == min(sol.get_var_solution(x[i]).get_start() for i in range(self.no_jobs)), "Job 0 does not start first."
        else:
            # end_times = [start_time + duration for start_time, duration in zip(start_times, self.durations)]
            # end_times = [start_time + duration for start_time, duration in zip(start_times, self.durations)]
            assert max([x['end'] for x in xs]) <= self.horizon, "Project completion time exceeds horizon."

            for i, job_successors in enumerate(self.successors):
                for successor in job_successors:
                    assert xs[i]['end'] <= xs[successor - 1]['start'], f"Job {i} ends after job {successor} starts."
                    
            assert xs[0]['start'] == min([x['start'] for x in xs]), "Job 0 does not start first."


        return True

    def visualize(self, jobs):
        # self.no_jobs = len(x)

        # if sol and visu.is_visu_enabled():
        #     visu.timeline('Solution SchedOptional', 0, 110)
        #     for job_number in range(self.no_jobs):
        #         visu.sequence(name=job_number)
        #         wt = sol.get_var_solution(x[job_number])
        #         if wt.is_present():
        #             if wt.get_start() != wt.get_end():
        #                 visu.interval(wt, "salmon", x[job_number].get_name())
        # visu.show()

        # Define the data for the Gantt chart
        end_times = [job['end'] for job in jobs]
        start_times = [job['start'] for job in jobs]
        task_names = [job['name'] for job in jobs]

        # Create the Gantt chart
        fig, ax = plt.subplots()
        for i in range(self.no_jobs):
            ax.broken_barh(
                [(start_times[i], end_times[i] - start_times[i])], (i, 1), facecolors='blue')
        ax.set_ylim(0, self.no_jobs)
        ax.set_xlim(0, max(end_times))
        ax.set_xlabel('Time')
        ax.set_yticks(range(self.no_jobs))
        ax.set_yticklabels(['Activity %d' % i for i in range(self.no_jobs)])
        ax.grid(True)
        plt.show()

        # Resource usage
        # https://ibmdecisionoptimization.github.io/docplex-doc/cp/visu.rcpsp.py.html

        if visu.is_visu_enabled():  # sol and 
            load = [CpoStepFunction()
                    for j in range(self.no_renewable_resources)]
            for i in range(self.no_jobs):
                itv = CpoIntervalVarSolution(
                    None, True, start_times[i], end_times[i], start_times[i] - end_times[i])
                for j in range(self.no_renewable_resources):
                    if 0 < self.requests[j][i]:
                        load[j].add_value(
                            itv.get_start(), itv.get_end(), self.requests[j][i])

            visu.timeline('Solution for RCPSP ')  # + filename)
            visu.panel('Tasks')
            for i in range(self.no_jobs):
                visu.interval(CpoIntervalVarSolution(None, True, start_times[i], end_times[i], start_times[i] - end_times[i]), i, task_names[i])
            for j in range(self.no_renewable_resources):
                visu.panel('R' + str(j+1))
                visu.function(
                    segments=[(0, 200, self.renewable_capacities[j])], style='area', color='lightgrey')
                visu.function(segments=load[j], style='area', color=j)
            visu.show()
