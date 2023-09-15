from src.general_optimization_solver import load_raw_instance
from src.mmrcpsp.solver import MMRCPSPSolver

import sys

instance_no = int(sys.argv[1])

if instance_no % 5 == 0:
    instance = instance_no // 5
    parameter = 5
else:
    instance = instance_no // 5 + 1
    parameter = instance_no % 5

print(f"raw_data/mm-rcpsp/MMLIB50/J50{instance}_{parameter}.mm")
instance = load_raw_instance(f"raw_data/mm-rcpsp/MMLIB50/J50{instance}_{parameter}.mm", "raw_data/mm-rcpsp/mmlib50_results.xlsx", "mmlib")
MMRCPSPSolver(no_workers=14).solve(instance)
instance.dump_json()
