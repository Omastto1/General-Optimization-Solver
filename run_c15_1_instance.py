from src.general_optimization_solver import load_raw_instance
from src.solvers.mmrcpsp import MMRCPSPSolver

import sys

# start with 1540_1
instance_no = int(sys.argv[1])

if instance_no % 10 == 0:
    instance = instance_no // 10 + 39
    parameter = 10
else:
    instance = instance_no // 10 + 40
    parameter = instance_no % 10


instance = load_raw_instance(f"raw_data/mm-rcpsp/c15.mm/c15{instance}_{parameter}.mm", "raw_data/mm-rcpsp/c15opt.mm.html", "c15")
MMRCPSPSolver(no_workers=14).solve(instance)
instance.dump_json()
