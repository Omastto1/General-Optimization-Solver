"""_summary_

Returns:
    _type_: _description_
"""
import numpy as np
import networkx as nx

from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

# from src.rcpsp.solvers.solver_cp import RCPSPCPSolver
# from src.rcpsp.solvers.solver_ga import RCPSPGASolver
from src.general_optimization_solver import load_raw_instance  # , load_instance, load_raw_benchmark
from src.common.solver import GASolver


def decode_chromosome(instance, chromosome):
    """Chromosome representation decoder from DOI: 10.1007/s10732-010-9142-2

    Args:
        instance (_type_): _description_
        chromosome (_type_): _description_

    Returns:
        _type_: _description_
    """
    priorities = []
    for i in range(instance.no_jobs):
        priority_adjustment = (1 + chromosome[i]) / 2
        priorities.append(
            instance.longest_length_paths[i] / instance.longest_length_paths[0] * priority_adjustment)
    priorities = chromosome[:instance.no_jobs]

    delays = chromosome[instance.no_jobs:] * 1.5 * max(instance.durations)

    return priorities, delays


def update_e(instance, S, F, g, t, delays):
    """return new E[g] element
    """
    finished_predecessors_jobs = [job_index for job_index in range(len(instance.predecessors)) if len(
        set([i - 1 for i in instance.predecessors[job_index]]).difference(S[g-1])) == 0]
    not_scheduled_jobs = [
        job_index for job_index in finished_predecessors_jobs if job_index not in S[g-1]]

    jobs_prec_feasible_in_delay_g = []
    for job_index in not_scheduled_jobs:
        all_passed = True
        for predecessor in instance.predecessors[job_index]:
            if F[predecessor - 1] > t[-1] + delays[g]:  # predecessors are 1-based
                all_passed = False
                break
        if all_passed:
            jobs_prec_feasible_in_delay_g.append(job_index)

    return jobs_prec_feasible_in_delay_g


# https://www.sciencedirect.com/science/article/pii/S0305054807001359
# jak muzu pristupovat ke gamma[g], kdyz je gamma na zacatku 1, ale inicializuju jen gamma[0]
# Jak muze byt gamma, S0, A0 atd na zacatku 0, kdyz potom pouzivam setove U ??
# Proc je na radku 4 // Update Eg, kdyz tam jen checkuju
# 7 ????????????
# 7 jak muzu pristupovat ke gamma[g], kdyz se updatuje az na radku 8
def construct_schedules(instance, E, S, F, t, priorities, delays):
    """E awaits [0, E1]

    Args:
        instance (_type_): _description_
        E (_type_): set responsible for forcing the selection to be ade only amongst activities which will have a delay smaller or equal to maximum allowed delay
        Pj - predecessor activities
        Fj - 
        F n+1 = max{F l |l ∈ Pn+1}. - The makespan of the solution is given by the maximum of all predecessor activities
            of activity n + 1
    """
    g = 1
    A = [[0], [0]]  # A0
    gamma = [np.array([0]), np.array([0])]  # gamma0 - jobs finish times
    EF = [0] + [None] * (instance.no_jobs - 1)  # earliest job finish times

    remaining_capacities = []
    for k in range(instance.no_renewable_resources):
        remaining_capacities.append(
            [instance.renewable_capacities[k]] * sum(instance.durations))

    # PROBABLY WRONG INDEX
    while len(S[g-1]) < instance.no_jobs:
        last_g = g
        while len(E[g]) > 0:
            # Extend t if outer while loop not reached in previous iteration
            # if len(t) != g:
            #     possible_times = gamma[g - 1][gamma[g - 1] > t[g - 1]]
            #     t.append(min(possible_times))

            max_prio_to_schedule = np.argmax(priorities[i] for i in E[g])
            j_star = E[g][max_prio_to_schedule]

            # successors and predecessors are 1-based
            max_pred_finish = max(F[i - 1] for i in instance.predecessors[j_star])  # pred 1-based

            EF[j_star] = max_pred_finish + instance.durations[j_star]

            # CHANGE: PAPER SHOWS gamma[g], but there is only gamma[0] initiated in first iteration g=1
            ts_temp = gamma[g-1][gamma[g-1] >= max_pred_finish]

            ts = []
            for t_temp in ts_temp:
                passed = True
                for greek in range(t_temp, t_temp + instance.durations[j_star]):
                    for k in range(instance.no_renewable_resources):
                        if instance.requests[k][j_star] > remaining_capacities[k][greek]:
                            passed = False
                            break
                    if not passed:
                        break

                if passed:
                    ts.append(t_temp)

            F[j_star] = min(ts) + instance.durations[j_star]

            # not relevant right now different than in paper, shifted after g increment
            #  Different than in paper, it updates S[g-1] (g before increment) and then checks g (g after increment) in while
            S.append(S[g-1] + [j_star])
            gamma[g] = np.append(gamma[g], F[j_star])
            gamma.append(gamma[g])

            g += 1

            A.append([j for j in range(instance.no_jobs) if F[j] is not None and F[j] - instance.durations[j] <= t[-1] < F[j]])
            jobs_prec_feasible_in_delay_g = update_e(
                instance, S, F, g, t, delays)
            E.append(jobs_prec_feasible_in_delay_g)

            # Update Rd[k][t]
            for i in range(F[j_star] - instance.durations[j_star], F[j_star]):
                for k in range(instance.no_renewable_resources):
                    remaining_capacities[k][i] -= instance.requests[k][j_star]
            # RDk (t g ) = R k (t g ) − ∑(j ∈Ag) r_j,k
            
            # for k in range(instance.no_renewable_resources):
            #     remaining_capacities[k][t[-1]] = instance.renewable_capacities[k] - sum(instance.requests[k][j] for j in A[g])

        # WRONG
        # t[g] = min( _t for _t in gamma[g-1] if _t > t[g-1])
        # while len(E[g]) == 0 and len(S[g-1]) < instance.no_jobs:
        #     t[g] += 1
        #     jobs_prec_feasible_in_delay_g = update_e(
        #         instance, S, F, g, t, delays)
        #     E[g] = jobs_prec_feasible_in_delay_g
        # _g = len(t)
        # while _g < g:
        if len(S[g-1]) != instance.no_jobs:
            possible_times = gamma[g][gamma[g] > t[-1]]
            t.append(min(possible_times))

            # _g += 1
        
        jobs_prec_feasible_in_delay_g = update_e(
            instance, S, F, g, t, delays)
        E[-1] = jobs_prec_feasible_in_delay_g
    
    # print(priorities[:5])
    # print(F)
    return F


def compute_modified_makespan(instance, F):
    """Compute modified makespan according to DOI: 10.1007/s10732-010-9142-2

    Args:
        instance (_type_): instance
        F (list[float]): list of jobs finish times

    Returns:
        float: modified makespan
    """
    makespan = max(F[i-1] for i in instance.predecessors[instance.no_jobs - 1])

    l_distance = 2
    jobs_in_l_distance_to_target = [i for i in range(
        instance.no_jobs) if 0 < instance.distances[i] <= l_distance]
    modified_makespan = makespan + sum(F[i] for i in jobs_in_l_distance_to_target) / (
        len(jobs_in_l_distance_to_target) * F[instance.no_jobs - 1])
    # print(makespan, end=" ")

    return modified_makespan


def fitness_func(instance, x, out):
    """Fitness func that is being fed to pymoo algorithm

    Args:
        instance (_type_): _description_
        x (list[float]): chromosome
        out (dict): pymoo specific output

    Returns:
        out (dict): pymoo specific output
    """
    priorities, delays = decode_chromosome(instance, x)

    S = [[0]]  # S0  - scheduled jobs
    # E = [[], [job_index for job_index in range(len(instance.predecessors)) if len(instance.predecessors[job_index]) == 0]]
    E = [[0]]
    t = [0, 0]  # t1
    # job finish times (in terms of precedence and capacity)
    F = [0] + [None] * (instance.no_jobs - 1)

    g = 1

    jobs_prec_feasible_in_delay_g = update_e(instance, S, F, g, t, delays)

    E.append(jobs_prec_feasible_in_delay_g)
    F = construct_schedules(instance, E, S, F, t, priorities, delays)

    modified_makespan = compute_modified_makespan(instance, F)

    out["F"] = modified_makespan
    # out["G"] = 0
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
                super().__init__(n_var=2*instance.no_jobs, n_obj=1,
                                 n_constr=0, xu=1, xl=0)
                self.instance = instance
                self.fitness_func = fitness_func_

            def _evaluate(self, x, out, *args, **kwargs):
                out = self.fitness_func(self.instance, x, out)

                assert "solution" not in out, "Do not use `solution` key, it is pymoo reserved keyword"

                return out

        problem = RCPSP(instance, self.fitness_func)
        res = minimize(problem, self.algorithm, self.termination,
                       verbose=True, seed=self.seed)
    
        if update_history:
            X = np.floor(res.X).astype(int)
            d = {}
            problem._evaluate(X, d)

            start_times = d['start_times']
            fitness_value = max(start_times[i] + instance.durations[i] for i in range(len(instance.durations)))
            export = {"tasks_schedule": [{"start": start_times[i], "end": start_times[i] + instance.durations[i], "name": f"Task_{i}"} for i in range(instance.no_jobs)]}

            fitness_value = int(fitness_value) # F - modified makespan (< 1)
            solution_info = f"start_times: {start_times}"
            self.add_run_to_history(instance, fitness_value, solution_info, exec_time=round(res.exec_time, 2))

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
    n_elites=40,
    n_offsprings=80,
    n_mutants=25,
    bias=0.7,
    eliminate_duplicates=MyElementwiseDuplicateElimination())


BRKGA_solver = RCPSPGASolver(
    algorithm, fitness_func, ("n_gen", 250), solver_name="BRKGA")  # , seed=1


instance_ = load_raw_instance("raw_data/rcpsp/j120.sm/j1201_1.sm", "")
# instance_ = load_raw_instance("raw_data/rcpsp/RG300/RG300_1.rcp", "")  # , "1Dbinpacking"


G = nx.DiGraph()
for job in range(instance_.no_jobs):
    G.add_node(job)
    for predecessor_ in instance_.predecessors[job]:
        G.add_edge(job, predecessor_ - 1)

instance_.distances = nx.single_source_bellman_ford_path_length(
    G, instance_.no_jobs - 1)

G2 = nx.DiGraph()
for job in range(instance_.no_jobs):
    G2.add_node(job)
    for predecessor_ in instance_.predecessors[job]:
        G2.add_edge(job, predecessor_ - 1, weight=-1)


longest_length_paths_negative = nx.single_source_bellman_ford_path_length(
    G2, instance_.no_jobs - 1)
instance_.longest_length_paths = {k: -v for k,
                                 v in longest_length_paths_negative.items()}

# brkga_fitness_value, brkga_assignment, brkga_solution = BRKGA_solver.solve(
#     instance_, visualize=True, validate=True)
