from src.general_optimization_solver import load_raw_instance
from src.rcpsp.solver import RCPSPSolver

import sys

instance_no = int(sys.argv[1])

instance = load_raw_instance(f"raw_data/rcpsp/CV/cv{instance_no}.rcp", "raw_data/rcpsp/CV.xlsx", "patterson")
RCPSPSolver(no_workers=14).solve(instance)
instance.dump_json()
