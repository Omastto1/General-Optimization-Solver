from src.general_optimization_solver import load_raw_instance
from src.rcpsp.solver import RCPSPSolver

import sys

instance_no = 360 + int(sys.argv[1])

instance = load_raw_instance(f"raw_data/rcpsp/DC2-npv75/rcpspdc{instance_no}.rcp", "raw_data/rcpsp/DC2-npv75.xlsx", "patterson")
RCPSPSolver(no_workers=14).solve(instance)
instance.dump_json()
