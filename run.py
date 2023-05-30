from src.parsers.j30 import load_j30
from src.parsers.patterson import load_patterson
from src.parsers.c15 import load_c15
from src.parsers.jobshop import load_jobshop

from src.general_optimization_solver import load_instance, load_raw_instance, load_benchmark, load_raw_benchmark

from src.solvers.rcpsp import RCPSPSolver
from src.solvers.mmrcpsp import MMRCPSPSolver
from src.solvers.jobshop import JobShopSolver

# instance = load_raw_instance("raw_data/rcpsp/CV/cv1.rcp", "raw_data/rcpsp/CV.xlsx", "patterson")
# solution, _ = RCPSPSolver(TimeLimit=10).solve(instance, "CP")  # TimeLimit=1

# instance = load_raw_instance("raw_data/rcpsp/DC1/mv1.rcp", "raw_data/rcpsp/DC1.xlsx", "patterson")

# solution, _ = RCPSPSolver(TimeLimit=3).solve(instance, "CP")  # TimeLimit=1

# benchmark = load_benchmark("data/RCPSP/j30")
# benchmark.solve(RCPSPSolver(1), force_dump=True)


benchmark = load_raw_benchmark("raw_data/rcpsp/DC1", "raw_data/rcpsp/DC1.xlsx", "patterson")
benchmark.solve(RCPSPSolver(1), force_dump=True)

# instance = load_raw_instance("raw_data/rcpsp/j30.sm/j3022_7.sm", "raw_data/rcpsp/j30opt.sm", "j30")
# RCPSPSolver(1).solve(instance)



# len([a for a in Path("data/RCPSP/j30").iterdir()])


import os
from pathlib import Path
import json

solved_optimally = 0
number_of_instances = 0
for file in Path("data/RCPSP/j30").iterdir():
    with open(file, "r", encoding="utf-8") as file:
        data = json.load(file)

        number_of_instances += 1
        if data["run_history"] and data["run_history"][-1]["solve_status"] == "Optimal":
            solved_optimally += 1

print(f"{solved_optimally} out of {number_of_instances} solved optimally")

