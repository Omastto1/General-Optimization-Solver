from src.general_optimization_solver import load_raw_instance, load_instance, load_raw_benchmark
from src.jobshop.solvers.solver_cp import JobShopCPSolver
from src.rcpsp.solvers.solver_cp import RCPSPCPSolver
from src.mmrcpsp.solvers.solver_cp import MMRCPSPCPSolver
from src.binpacking1d.solvers.solver_cp import BinPacking1DCPSolver
from src.binpacking2d.solvers.solver_cp import BinPacking2DCPSolver
from src.strippacking2d.solvers.solver_cp_oriented import StripPacking2DCPSolver
from src.strippacking2d.solvers.solver_cp_not_oriented import StripPacking2DCPSolver as StripPacking2DCPSolverNotOriented
from src.strippacking2d.solvers.solver_cp_leveled import StripPackingLeveled2DCPSolver


## python -m examples.example_rcpsp

solvers_config = {"TimeLimit": 1}
instances = {
    # "jobshop": {"solver": JobShopCPSolver(**solvers_config),"path": "raw_data/jobshop/jobshop/abz5"},
    # "rcpsp": {"solver": RCPSPCPSolver(**solvers_config),"path": "raw_data/rcpsp/CV/cv1.rcp"},
    # "mmrcpsp": {"solver": MMRCPSPCPSolver(**solvers_config),"path": "raw_data/mm-rcpsp/c15.mm/c154_3.mm"},
    # "1dbinpacking": {"solver": BinPacking1DCPSolver(**solvers_config),"path": "raw_data/1d-binpacking/scholl_bin1data/N1C1W1_A.BPP"},
    # 2d bin packing not working - slow model build
    # "2dbinpacking": {"solver": BinPacking2DCPSolver(**solvers_config),"path": "raw_data/2d-binpacking/ngcut_bin/ngcut_1.BPP"},
    # 2d strip packing not working - slow model build
    # "strippacking": {"solver": StripPacking2DCPSolver(**solvers_config),"path": "raw_data/2d_strip_packing/benchmark/BENG01.TXT"},
    # "strippacking_not_oriented": {"solver": StripPacking2DCPSolverNotOriented(**solvers_config),"path": "raw_data/2d_strip_packing/benchmark/BENG01.TXT"},
    "strippacking_leveled": {"solver": StripPackingLeveled2DCPSolver(**solvers_config),"path": "raw_data/2d_strip_packing/benchmark/BENG01.TXT"}
}


for problem_name, problem_config in instances.items():
    print("Solving", problem_name)
    instance = load_raw_instance(problem_config["path"], "")
    problem_config["solver"].solve(instance, validate=True, visualize=True, force_execution=True)
