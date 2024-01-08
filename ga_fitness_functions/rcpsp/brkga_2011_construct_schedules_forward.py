"""_summary_

Returns:
    _type_: _description_
"""
import numpy as np

from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from src.general_optimization_solver import load_raw_instance, load_instance, load_raw_benchmark
from src.common.solver import GASolver



def update_D(instance, S, g):
    """return new E[g] element
    """
    finished_predecessors_jobs = [job_index for job_index in range(instance.no_jobs) if len(
        set([i - 1 for i in instance.predecessors[job_index]]).difference(S[g-1])) == 0]
    not_scheduled_jobs = [
        job_index for job_index in finished_predecessors_jobs if job_index not in S[g-1]]

    return not_scheduled_jobs

def construct_schedules_forward(instance, S, F, priorities):
    """E awaits [0, E1]

    Args:
        instance (_type_): _description_
        E (_type_): set responsible for forcing the selection to be ade only amongst activities which will have a delay smaller or equal to maximum allowed delay
        Pj - predecessor activities
        Fj - 
        F n+1 = max{F l |l ∈ Pn+1}. - The makespan of the solution is given by the maximum of all predecessor activities
            of activity n + 1
    """
    EF = [0] + [None] * (instance.no_jobs - 1)  # earliest job finish times

    remaining_capacities = []
    for k in range(instance.no_renewable_resources):
        remaining_capacities.append(
            [instance.renewable_capacities[k]] * sum(instance.durations))

    gamma = [np.array([], dtype=int)] + [None] * (instance.no_jobs - 1)   # gamma0 - jobs finish times
    j_star = 0
    D = [0]

    for g in range(1, instance.no_jobs):
        D.append(update_D(instance, S, g))

        gamma[g] = np.append(gamma[g-1], F[j_star])        

        max_prio_to_schedule = np.argmax([priorities[i] for i in D[g]])
        j_star = D[g][max_prio_to_schedule]

        # successors and predecessors are 1-based
        max_pred_finish = max(F[i - 1] for i in instance.predecessors[j_star])  # pred 1-based
        EF[j_star] = max_pred_finish + instance.durations[j_star]

        ts_temp = gamma[g][gamma[g] >= max_pred_finish]
        
        t_temp = min(ts_temp)
        while True:
            resource_violation = False
            for greek in range(t_temp, t_temp + instance.durations[j_star]):
                for k in range(instance.no_renewable_resources):
                    if instance.requests[k][j_star] > remaining_capacities[k][greek]:
                        resource_violation = True
                        break
                if resource_violation:
                    break

            if not resource_violation:
                break
            t_temp += 1

        
        for t in range(t_temp, t_temp + instance.durations[j_star]):
            for k in range(instance.no_renewable_resources):
                remaining_capacities[k][t] -= instance.requests[k][j_star]

        F[j_star] = t_temp + instance.durations[j_star]
        S.append(S[g-1] + [j_star])
    
    return F, remaining_capacities


def calculate_backward_pass_with_resources(tasks, project_duration, resources):
    # Initialize Latest Start Time (LST) and Latest Finish Time (LFT) for each task
    lst = {task: None for task in tasks}
    lft = {task: None for task in tasks}

    # Initialize resource availability at each time step
    resource_availability = {time: resources.copy() for time in range(project_duration + 1)}

    # Set the LFT for the last tasks (no subsequent tasks) to the project duration
    for task in tasks:
        if not tasks[task]['subsequent_tasks']:
            lft[task] = project_duration
            lst[task] = lft[task] - tasks[task]['duration']

    # Backward pass
    for task in sorted(tasks.keys(), reverse=True):
        if lft[task] is None:
            lft_candidates = [lst[subsequent_task] for subsequent_task in tasks[task]['subsequent_tasks']]
            lft[task] = min(lft_candidates) if lft_candidates else project_duration
        
        # Adjust LFT based on resource availability
        for time in range(lft[task] - tasks[task]['duration'], lft[task]):
            while not all(resource_availability[time][res] >= tasks[task]['resources'][res] for res in tasks[task]['resources']):
                lft[task] -= 1
                time -= 1
                if time < tasks[task]['duration']:
                    raise Exception("Unable to schedule task due to resource constraints")
        
        lst[task] = lft[task] - tasks[task]['duration']
        # Reserve resources
        for time in range(lst[task], lft[task]):
            for res in tasks[task]['resources']:
                resource_availability[time][res] -= tasks[task]['resources'][res]

    return lst, lft

# Example tasks with duration, resources required, subsequent tasks, and preceding tasks
tasks = {
    'A': {'duration': 4, 'resources': {'Labor': 2, 'Machine': 1}, 'subsequent_tasks': ['B', 'C'], 'preceding_tasks': []},
    'B': {'duration': 3, 'resources': {'Labor': 1, 'Machine': 2}, 'subsequent_tasks': ['D'], 'preceding_tasks': ['A']},
    'C': {'duration': 2, 'resources': {'Labor': 3, 'Machine': 1}, 'subsequent_tasks': ['D'], 'preceding_tasks': ['A']},
    'D': {'duration': 5, 'resources': {'Labor': 2, 'Machine': 1}, 'subsequent_tasks': [], 'preceding_tasks': ['B', 'C']}
}

resources = {'Labor': 5, 'Machine': 3}  # Total available resources
project_duration = 14  # Total duration of the project
lst, lft = calculate_backward_pass_with_resources(tasks, project_duration, resources)

for task in lst:
    print(f"Task {task}: LST = {lst[task]}, LFT = {lft[task]}")





def fitness_func(instance, x, out):
    """Fitness func that is being fed to pymoo algorithm

    Args:
        instance (_type_): _description_
        x (list[float]): chromosome
        out (dict): pymoo specific output

    Returns:
        out (dict): pymoo specific output
    """
    priorities = x

    S = [[0]]  # S0  - scheduled jobs
    F = [0] + [None] * (instance.no_jobs - 1)

    F = construct_schedules_forward(instance, S, F, priorities)

    
    makespan = max(F[i-1] for i in instance.predecessors[instance.no_jobs - 1])

    out["F"] = makespan
    out["G"] = [0] * instance.no_renewable_resources
    out["start_times"] = [F[i] - instance.durations[i] for i in range(len(instance.durations))]

    return out


class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return (a.X.round(2) == b.X.round(2)).all()


class RCPSPGASolver(GASolver):
    """GA SOLVER WRAPPER CLASS
    """
    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True):
        class RCPSP(ElementwiseProblem):
            """pymoo wrapper class
            """

            def __init__(self, instance, fitness_func_):
                super().__init__(n_var=instance.no_jobs, n_obj=1,
                                 n_constr=0, xu=1, xl=0)
                self.instance = instance
                self.fitness_func = fitness_func_

            def _evaluate(self, x, out, *args, **kwargs):
                out = self.fitness_func(self.instance, x, out)

                assert "solution" not in out, "Do not use `solution` key, it is pymoo reserved keyword"

                return out
        raise Exception("Import Solver from src/rcpsp/solvers/ga_solve.py")

        problem = RCPSP(instance, self.fitness_func)
        res = minimize(problem, self.algorithm, self.termination,
                       verbose=True, seed=self.seed,
                       callback=self.callback)
    
        if update_history:
            X = np.floor(res.X).astype(int)
            d = {}
            problem._evaluate(X, d)

            start_times = d['start_times']
            fitness_value = max(start_times[i] + instance.durations[i] for i in range(len(instance.durations)))
            export = {"tasks_schedule": [{"start": start_times[i], "end": start_times[i] + instance.durations[i], "name": f"Task_{i}"} for i in range(instance.no_jobs)]}

            fitness_value = int(fitness_value) # F - modified makespan (< 1)
            solution_info = f"start_times: {start_times}"
            solution_progress = res.algorithm.callback.data['progress']
            self.add_run_to_history(instance, fitness_value, solution_info, solution_progress, exec_time=round(res.exec_time, 2))

        # if res.F is not None:
        #     X = np.floor(res.X).astype(int)
        #     fitness_value = res.F[0]
        #     print('Objective value:', fitness_value)

        #     d = {}
        #     problem._evaluate(X, d)
        #     start_times = d['start_times']
        #     export = {"tasks_schedule": [{"start": start_times[i], "end": start_times[i] +
        #                                   instance.durations[i], "name": f"Task_{i}"} for i in range(instance.no_jobs)]}

        # if res.F is not None:
        #     return fitness_value, start_times, res
        # else:
        return None, None, res


# values from https://pymoo.org/algorithms/soo/brkga.html
algorithm = BRKGA(
    n_elites=10,
    n_offsprings=20,
    n_mutants=8,
    bias=0.7,
    eliminate_duplicates=MyElementwiseDuplicateElimination())


if __name__ == "__main__":

    BRKGA_solver = RCPSPGASolver(
        algorithm, fitness_func, ("n_gen", 250), solver_name="BRKGA")  # , seed=1
    

    instance_ = load_raw_instance("raw_data/rcpsp/j30.sm/j3010_3.sm", "")
    # benchmark = load_raw_benchmark("raw_data/rcpsp/j30.sm/", no_instances=100)
    # instance_ = load_raw_instance("raw_data/rcpsp/RG300/RG300_1.rcp", "")  # , "1Dbinpacking"


    from pymoo.termination.ftol import SingleObjectiveSpaceTermination
    from pymoo.termination.robust import RobustTermination
    term = RobustTermination(SingleObjectiveSpaceTermination(tol = 0.1), period=30)
    BRKGA_solver = RCPSPGASolver(algorithm, fitness_func, term, seed=1, solver_name="BRKGA_rcpsp_15_57_18_0.7")

    BRKGA_solver.solve(instance_, visualize=False, validate=True, force_dump=False)

    benchmark.generate_solver_comparison_percent_deviation_markdown_table()