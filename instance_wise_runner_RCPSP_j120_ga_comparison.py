
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA, comp_by_cv_and_fitness
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.termination import get_termination
from pymoo.operators.selection.tournament import TournamentSelection

from pymoo.algorithms.soo.nonconvex.brkga import BRKGA

from src.rcpsp.solvers.solver_ga import RCPSPGASolver
from src.general_optimization_solver import load_raw_instance, load_instance

# from ga_fitness_functions.rcpsp.naive_backward import fitness_func_backward
# from ga_fitness_functions.rcpsp.naive_forward import fitness_func_forward

from examples.brkga_2011_construct_schedules_forward import fitness_func as bkrga_fitness_func  #  RCPSPGASolver as PaperRCPSPGASolver,

no_eval = 5000
term_eval = get_termination("n_eval", no_eval)


class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return (a.X.astype(int) == b.X.astype(int)).all()
    

d = {
# "GA": [{
#     "name": "ga_half",
#     "pop_size": 60,
#     "n_offsprings": 60,
#     "cross_prob": 0.9,
#     "mut_eta": 30
# },{
#     "name": "ga",
#     "pop_size": 120,
#     "n_offsprings": 120,
#     "cross_prob": 0.9,
#     "mut_eta": 30
# },{
#     "name": "ga_double",
#     "pop_size": 240,
#     "n_offsprings": 240,
#     "cross_prob": 0.9,
#     "mut_eta": 30
# }],
    "BRKGA": [
        {
    "name": "brkga_TOP_25%_BOT_20%",
    "n_elites": 30,
    "n_offsprings": 66,
    "n_mutants": 24,
    "bias": 0.7
},
{
    "name": "brkga_TOP_25%_BOT_15%",
    "n_elites": 30,
    "n_offsprings": 72,
    "n_mutants": 18,
    "bias": 0.7
},
# {
#     "name": "brkga_TOP_20%",
#     "n_elites": 24,
#     "n_offsprings": 78,
#     "n_mutants": 18,
#     "bias": 0.7
# },{
#     "name": "brkga_TOP_10%",
#     "n_elites": 12,
#     "n_offsprings": 72,
#     "n_mutants": 36,
#     "bias": 0.7
# },{
#     "name": "brkga_TOP_15%",
#     "n_elites": 18,
#     "n_offsprings": 76,
#     "n_mutants": 26,
#     "bias": 0.7
# }
]}

problem_type = "RCPSP"
benchmark_name = "j120.sm"


id = int(os.environ['SLURM_ARRAY_TASK_ID'])

print("JOB ID")
print(id)

parameter = id // 10 + 1
instance = id % 10

# ranges from j601_1.sm to j60_10.sm
if id % 10 == 0:
    parameter -= 1
    instance = 10

# SPECIFIC BENCHMARK INSTANCE
# instance = load_raw_instance(f"raw_data/{problem_type.lower()}/{benchmark_name}/{benchmark_name.split('.')[0]}{parameter}_{instance}.sm")
instance = load_instance(f"master_thesis_data_ga_comp_test/{problem_type}/{benchmark_name}/{benchmark_name.split('.')[0]}{parameter}_{instance}.json")

# for algorithm_type, algorithm_variants in d.items():
#     for algorithm_config in algorithm_variants:
#         if algorithm_type == "GA":
#             algorithm = GA(
#                 pop_size=algorithm_config["pop_size"],
#                 n_offsprings=algorithm_config["n_offsprings"],
#                 sampling=FloatRandomSampling(),
#                 crossover=TwoPointCrossover(prob=algorithm_config['cross_prob']),
#                 mutation=PolynomialMutation(eta=algorithm_config['mut_eta']),
#                 selection=TournamentSelection(comp_by_cv_and_fitness),
#                 eliminate_duplicates=MyElementwiseDuplicateElimination()
#             )

#             solver_name = f"{algorithm_config['name']} {algorithm_config['pop_size']}_{algorithm_config['n_offsprings']}_{algorithm_config['cross_prob']}_{algorithm_config['mut_eta']}_{no_eval}evals"

#         elif algorithm_type == "BRKGA":
#             algorithm = BRKGA(
#                 n_elites=algorithm_config["n_elites"],
#                 n_offsprings=algorithm_config["n_offsprings"],
#                 n_mutants=algorithm_config["n_mutants"],
#                 bias=algorithm_config["bias"],
#                 eliminate_duplicates=MyElementwiseDuplicateElimination()
#             )

#             solver_name = f"{algorithm_config['name']} {algorithm_config['n_elites']}_{algorithm_config['n_offsprings']}_{algorithm_config['n_mutants']}_{algorithm_config['bias']}_{no_eval}evals"
            
#         else:
#             raise ValueError("Wrong algorithm type, insert one of 'GA' or 'BRKGA'")
        
        
#         solver = RCPSPGASolver(
#             algorithm, bkrga_fitness_func, term_eval, seed=1, solver_name=solver_name)
#         solver.solve(instance, validate=True, force_execution=True, force_dump=False)

from pymoo.operators.crossover.pntx import SinglePointCrossover, TwoPointCrossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.hux import HalfUniformCrossover

for crossover in [SinglePointCrossover, TwoPointCrossover, SBX, UniformCrossover, HalfUniformCrossover]:
    algorithm = GA(
        pop_size=120,
        n_offsprings=120,
        sampling=FloatRandomSampling(),
        crossover=crossover(),
        mutation=PolynomialMutation(eta=30),
        selection=TournamentSelection(comp_by_cv_and_fitness),
        eliminate_duplicates=MyElementwiseDuplicateElimination()
    )

    print(f"running {algorithm.mating.crossover.__class__.__name__}")

    solver_name = f"GA 120_120_{algorithm.mating.crossover.__class__.__name__}_30_{no_eval}evals"
    solver = RCPSPGASolver(
        algorithm, bkrga_fitness_func, term_eval, seed=1, solver_name=solver_name)
    solver.solve(instance, validate=True, force_execution=True, force_dump=False)


instance.dump(dir_path=f"master_thesis_data_ga_comp_test/{problem_type}/{benchmark_name}")
