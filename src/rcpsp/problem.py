from src.common.optimization_problem import OptimizationProblem
import docplex.cp.utils_visu as visu
import matplotlib.pyplot as plt
from docplex.cp.model import CpoStepFunction
from docplex.cp.solution import CpoIntervalVarSolution


def create_predecessors(successors_list):
    predecessors = {}

    # Fix variable name conflict by using a different name for loop variable
    # successors are indexed starting from zero
    for job_index, successors in enumerate(successors_list, start=1):
        for successor in successors:
            if successor not in predecessors:
                predecessors[successor] = []
            predecessors[successor].append(job_index)

    # Create a list of predecessors for each job
    predecessors_list = []
    for job_index in range(len(successors_list)):
        predecessors_list.append(predecessors.get(job_index + 1, []))

    return predecessors_list


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
        self.predecessors = create_predecessors(self.successors)

        # available resource capacity
        self.renewable_capacities = self._data["resources"]["renewable_resources"]["renewable_availabilities"]
        self.requests = [[self._data["job_specifications"][i]["modes"][0]["request_duration"]
                          [f"R{k+1}"] for i in range(self.no_jobs)] for k in range(self.no_renewable_resources)]

    def validate(self, sol, x, model_variables_export=None):  #
        """ xs holding arrays of "start", "end" dictionaries """
        # TODO:
        if sol is not None:
            # Find the max time to define the range for checking
            max_time = max(sol.get_var_solution(
                x[i]).get_end() for i in range(self.no_jobs))

            # Check each time unit
            for k in range(self.no_renewable_resources):
                for time in range(max_time):
                    if self.__class__.__name__ == "RCPSP":
                        resource_used = sum(self.requests[i][k] for i in range(self.no_jobs) if sol.get_var_solution(
                            x[i]).get_start() <= time < sol.get_var_solution(x[i]).get_end())
                    elif self.__class__.__name__ == "MMRCPSP":
                        resource_used = sum(self.requests[k][i][sol.get_var_solution(x[i]).get_name().split("_")[-1]] for i in range(
                            self.no_jobs) if sol.get_var_solution(x[i]).get_start() <= time < sol.get_var_solution(x[i]).get_end())

                    assert resource_used <= self.renewable_capacities[
                        k], f"More resource used than renewable capacity {k} has available."

            assert sol.get_objective_value(
            ) <= self.horizon, "Project completion time exceeds horizon."

            for i, job_successors in enumerate(self.successors):
                for successor in job_successors:
                    assert sol.get_var_solution(x[i]).get_end() <= sol.get_var_solution(
                        x[successor - 1]).get_start(), f"Job {i} ends after job {successor} starts."

            assert sol.get_var_solution(x[0]).get_start() == min(sol.get_var_solution(
                x[i]).get_start() for i in range(self.no_jobs)), "Job 0 does not start first."
        else:
            xs = model_variables_export['tasks_schedule']

            # Find the max time to define the range for checking
            max_time = max(xs[i]['end'] for i in range(self.no_jobs))

            # Check each time unit
            for k in range(self.no_renewable_resources):
                for time in range(max_time):
                    if self.__class__.__name__ == "RCPSP":
                        resource_used = sum(self.requests[k][i] for i in range(
                            self.no_jobs) if xs[i]['start'] <= time < xs[i]['end'])
                    elif self.__class__.__name__ == "MMRCPSP":
                        resource_used = sum(self.requests[k][i][xs[i]['mode']] for i in range(
                            self.no_jobs) if xs[i]['start'] <= time < xs[i]['end'])

                    assert resource_used <= self.renewable_capacities[
                        k], f"More resource used than renewable capacity {k} has available."

            assert max([x['end'] for x in xs]
                       ) <= self.horizon, "Project completion time exceeds horizon."

            for i, job_successors in enumerate(self.successors):
                for successor in job_successors:
                    assert xs[i]['end'] <= xs[successor -
                                              1]['start'], f"Job {i} ends after job {successor} starts."

            assert xs[0]['start'] == min(
                [x['start'] for x in xs]), "Job 0 does not start first."

        return True

    def visualize(self, model_variables_export):
        jobs = model_variables_export['tasks_schedule']

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
                visu.interval(CpoIntervalVarSolution(
                    None, True, start_times[i], end_times[i], start_times[i] - end_times[i]), i, task_names[i])
            for j in range(self.no_renewable_resources):
                visu.panel('R' + str(j+1))
                visu.function(segments=load[j], style='area', color=j)
            visu.show()
