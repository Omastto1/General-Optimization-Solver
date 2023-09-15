import json
from pathlib import Path

from src.mmrcpsp.c15 import load_c15, load_c15_solution
from src.rcpsp.j30 import load_j30, load_j30_solution
from src.rcpsp.patterson import load_patterson , load_patterson_solution
from src.jobshop.parser import load_jobshop, load_jobshop_solution
from src.strippacking2d.parser import load_strip_packing, load_strip_packing_solution
from src.mmrcpsp.mmlib import load_mmlib, load_mmlib_solution

from src.common.optimization_problem import Benchmark
from src.mmrcpsp.problem import MMRCPSP
from src.rcpsp.problem import RCPSP
from src.jobshop.problem import JobShop
from src.strippacking2d.problem import StripPacking2D


def load_raw_benchmark(directory_path, solution_path, format=None, no_instances=0, force_dump=True):
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)

    print("Loading raw benchmark data")
    benchmark_instances = {}
    for i, instance in enumerate(directory_path.iterdir()):
        if no_instances > 0 and i >= no_instances:
            break

        if instance.is_file():
            print(f"loading {instance}")
            instance_path = str(instance).replace("\\", "/")
            instance_name = instance_path.split("/")[-1].split(".")[0]
            instance = load_raw_instance(instance_path, solution_path, format)

            benchmark_instances[instance_name] = instance
    
    benchmark = Benchmark("test benchmark", benchmark_instances)

    if force_dump:
        benchmark.dump()

    return benchmark


def load_benchmark(directory_path, no_instances=0):
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)

    benchmark_instances = {}
    for instance in directory_path.iterdir():
        if no_instances > 0 and i >= no_instances:
            break

        if instance.is_file():
            instance_path = str(instance).replace("\\", "/")
            # print(instance)
            # print(instance_path)
            instance_name = instance_path.split("/")[-1].split(".")[0]
            instance = load_instance(instance_path)

            benchmark_instances[instance_name] = instance
    
    benchmark = Benchmark("test benchmark", benchmark_instances)
    return benchmark

def load_raw_instance(path, solution_path, format, verbose=False):
    benchmark_name = path.split("/")[-2].split(".")[0]
    instance_name = path.split("/")[-1].split(".")[0]

    assert format is not None, "Specify valid raw data input argument `format`"
    
    if format == "c15":
        data = load_c15(path, verbose)
        solution = load_c15_solution(solution_path, benchmark_name, instance_name)

        instance = MMRCPSP(benchmark_name, instance_name, data, solution, [])
    elif format == "j30":
        data = load_j30(path, verbose)
        solution = load_j30_solution(solution_path, benchmark_name, instance_name)

        instance = RCPSP(benchmark_name, instance_name, data, solution, [])
    elif format == "patterson":
        print(solution_path, instance_name)
        data = load_patterson(path, verbose)
        solution = load_patterson_solution(solution_path, instance_name)

        instance = RCPSP(benchmark_name, instance_name, data, solution, [])
    elif format == "jobshop":
        data = load_jobshop(path, verbose)
        solution = load_jobshop_solution(solution_path, instance_name)

        instance = JobShop(benchmark_name, instance_name, data, solution, [])
    elif format == "strippacking":
        data = load_strip_packing(path, verbose)
        solution = load_strip_packing_solution(solution_path, instance_name)

        instance = StripPacking2D(benchmark_name, instance_name, data, solution, [])
    elif format == "mmlib":
        data = load_mmlib(path, verbose)
        solution = load_mmlib_solution(solution_path, instance_name)

        instance = MMRCPSP(benchmark_name, instance_name, data, solution, [])
    
    return instance


def load_instance(path):
    assert path.endswith(".json"), "Invalid input file, should be .json with predefined structure"
    
    with open(path, "r") as f:
        instance_dict = json.load(f)
    
    benchmark_name = instance_dict["benchmark_name"]
    instance_name = instance_dict["instance_name"]
    instance_kind = instance_dict["instance_kind"]
    data = instance_dict["data"]
    solution = instance_dict["reference_solution"]
    run_history = instance_dict["run_history"]

    if instance_kind == "MMRCPSP":
        instance = MMRCPSP(benchmark_name, instance_name, data, solution, run_history)
    elif instance_kind == "RCPSP":
        instance = RCPSP(benchmark_name, instance_name, data, solution, run_history)
    elif instance_kind == "JOBSHOP":
        instance = JobShop(benchmark_name, instance_name, data, solution, run_history)
    elif instance_kind == "2DSTRIPPACKING":
        instance = StripPacking2D(benchmark_name, instance_name, data, solution, run_history)
    else:
        raise ValueError("Invalid instance kind, should be one of: MMRCPSP, RCPSP, JOBSHOP")

    return instance