from src.general_optimization_solver import load_raw_instance
from src.rcpsp.solver import RCPSPSolver

import sys

instance_no = int(sys.argv[1])

if instance_no % 10 == 0:
    instance = instance_no // 10
    parameter = 10
else:
    instance = instance_no // 10 + 1
    parameter = instance_no % 10

instance = load_raw_instance(f"raw_data/rcpsp/j30.sm/j30{instance}_{parameter}.sm", "raw_data/rcpsp/j30opt.sm", "j30")
RCPSPSolver(no_workers=14).solve(instance)
instance.dump_json()
