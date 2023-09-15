from src.rcpsp.j30 import load_j30
from src.rcpsp.patterson import load_patterson
from src.mmrcpsp.c15 import load_c15
from src.jobshop.parser import load_jobshop

from src.general_optimization_solver import load_instance, load_raw_instance, load_benchmark, load_raw_benchmark

from src.rcpsp.solver import RCPSPSolver
from src.mmrcpsp.solver import MMRCPSPSolver
from src.jobshop.solver import JobShopSolver

# instance = load_raw_instance("raw_data/rcpsp/CV/cv1.rcp", "raw_data/rcpsp/CV.xlsx", "patterson")
# solution, _ = RCPSPSolver(TimeLimit=10).solve(instance, "CP")  # TimeLimit=1

instance = load_raw_instance("raw_data/rcpsp/CV/cv1.rcp", "raw_data/rcpsp/CV.xlsx", "patterson")

RCPSPSolver(3).solve(instance)

# solution, _ = RCPSPSolver(TimeLimit=3).solve(instance, "CP")  # TimeLimit=1

# benchmark = load_benchmark("data/RCPSP/j30")
# benchmark.solve(RCPSPSolver(1), force_dump=True)


# benchmark = load_raw_benchmark("raw_data/rcpsp/DC1", "raw_data/rcpsp/DC1.xlsx", "patterson")
# benchmark.solve(RCPSPSolver(1), force_dump=True)

# benchmark = load_raw_benchmark("raw_data/jobshop/jobshop", "raw_data/jobshop/instances_results.txt", "jobshop")
# benchmark.solve(JobShopSolver(1), force_dump=True)

# benchmark = load_raw_benchmark("raw_data/jobshop/jobshop", "raw_data/jobshop/instances_results.txt", "jobshop")
# benchmark = load_raw_benchmark("raw_data/mm-rcpsp/c15.mm", "raw_data/mm-rcpsp/c15opt.mm.html", "c15")
# benchmark = load_raw_benchmark("raw_data/mm-rcpsp/c21.mm", "raw_data/mm-rcpsp/c21opt.mm.html", "c15")
# benchmark = load_raw_benchmark("raw_data/mm-rcpsp/j10.mm", "raw_data/mm-rcpsp/j10opt.mm.html", "c15")
# benchmark = load_raw_benchmark("raw_data/rcpsp/j30.sm", "raw_data/rcpsp/j30opt.sm", "j30")
# benchmark = load_raw_benchmark("raw_data/rcpsp/j60.sm", "raw_data/rcpsp/j60lb.sm", "j30")
# benchmark = load_raw_benchmark("raw_data/rcpsp/j90.sm", "raw_data/rcpsp/j90lb.sm", "j30")
# benchmark = load_raw_benchmark("raw_data/rcpsp/j120.sm", "raw_data/rcpsp/j120lb.sm", "j30")
# benchmark = load_raw_benchmark("raw_data/rcpsp/DC1", "raw_data/rcpsp/DC1.xlsx", "patterson")
# benchmark = load_raw_benchmark("raw_data/rcpsp/DC2-npv25", "raw_data/rcpsp/DC2-npv25.xlsx", "patterson")
# benchmark = load_raw_benchmark("raw_data/rcpsp/DC2-npv50", "raw_data/rcpsp/DC2-npv50.xlsx", "patterson")
# benchmark = load_raw_benchmark("raw_data/rcpsp/DC2-npv75", "raw_data/rcpsp/DC2-npv75.xlsx", "patterson")
# benchmark = load_raw_benchmark("raw_data/rcpsp/DC2-npv100", "raw_data/rcpsp/DC2-npv100.xlsx", "patterson")
# benchmark = load_raw_benchmark("raw_data/rcpsp/CV", "raw_data/rcpsp/CV.xlsx", "patterson")
# benchmark = load_raw_benchmark("raw_data/rcpsp/1kNetRes", "raw_data/rcpsp/1kNetRes.xlsx", "patterson")
# benchmark = load_raw_benchmark("raw_data/rcpsp/ResSet (for NetRes)", "raw_data/rcpsp/ResSet (for NetRes).xlsx", "patterson")
# benchmark = load_raw_benchmark("raw_data/rcpsp/RG30_Set 1", "raw_data/rcpsp/RG30_Set 1.xlsx", "patterson")
# benchmark = load_raw_benchmark("raw_data/rcpsp/RG30_Set 2", "raw_data/rcpsp/RG30_Set 2.xlsx", "patterson")
# benchmark = load_raw_benchmark("raw_data/rcpsp/RG30_Set 3", "raw_data/rcpsp/RG30_Set 3.xlsx", "patterson")
# benchmark = load_raw_benchmark("raw_data/rcpsp/RG30_Set 4", "raw_data/rcpsp/RG30_Set 4.xlsx", "patterson")
# benchmark = load_raw_benchmark("raw_data/rcpsp/RG30_Set 5", "raw_data/rcpsp/RG30_Set 5.xlsx", "patterson")

# missing reference
# benchmark = load_raw_benchmark("raw_data/rcpsp/MT30", "raw_data/rcpsp/MT30.xlsx", "patterson")
# benchmark = load_raw_benchmark("raw_data/rcpsp/RG300", , "patterson")
# benchmark = load_raw_benchmark("raw_data/rcpsp/sD", "raw_data/rcpsp/sD.xlsx", "patterson")

# instance = load_raw_instance("raw_data/rcpsp/j30.sm/j3022_7.sm", "raw_data/rcpsp/j30opt.sm", "j30")
# RCPSPSolver(1).solve(instance)



# len([a for a in Path("data/RCPSP/j30").iterdir()])


# import os
# from pathlib import Path
# import json

# solved_optimally = 0
# number_of_instances = 0
# for file in Path("data/JOBSHOP/jobshop").iterdir():
#     with open(file, "r", encoding="utf-8") as file:
#         data = json.load(file)

#         number_of_instances += 1
#         if data["run_history"] and data["run_history"][-1]["solve_status"] == "Optimal":
#             solved_optimally += 1

# print(f"{solved_optimally} out of {number_of_instances} solved optimally")

