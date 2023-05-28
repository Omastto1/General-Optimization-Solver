import json
from pathlib import Path

from src.parsers.c15 import load_c15, load_c15_solution
from src.parsers.j30 import load_j30, load_j30_solution
from src.parsers.patterson import load_patterson # , load_patterson_solution
from src.parsers.jobshop import load_jobshop, load_jobshop_solution
from docplex.cp.model import CpoModel

from src.mm_rcpsp import MMRCPSP
from src.rcpsp import RCPSP
from src.jobshop import JobShop
from src.optimization_problem import Benchmark



def load_raw_benchmark(directory_path, solution_path, format=None, force_dump=True):
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)

    print("Loading raw benchmark data")
    benchmark_instances = {}
    for instance in directory_path.iterdir():
        if instance.is_file():
            instance_path = str(instance).replace("\\", "/")
            instance_name = instance_path.split("/")[-1].split(".")[0]
            instance = load_raw_instance(instance_path, solution_path, format)

            benchmark_instances[instance_name] = instance
    
    benchmark = Benchmark("test benchmark", benchmark_instances)

    if force_dump:
        benchmark.dump()

    return benchmark


def load_benchmark(directory_path):
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)

    benchmark_instances = {}
    for instance in directory_path.iterdir():
        if instance.is_file():
            instance_path = str(instance).replace("\\", "/")
            # print(instance)
            # print(instance_path)
            instance_name = instance_path.split("/")[-1].split(".")[0]
            instance = load_instance(instance_path)

            benchmark_instances[instance_name] = instance
    
    benchmark = Benchmark("test benchmark", benchmark_instances)
    return benchmark

def load_raw_instance(path, solution_path, format):
    benchmark_name = path.split("/")[-2]
    instance_name = path.split("/")[-1].split(".")[0]

    assert format is not None, "Specify valid raw data input argument `format`"
    
    if format == "c15":
        data = load_c15(path)
        solution = load_c15_solution(solution_path, instance_name)

        instance = MMRCPSP(benchmark_name, instance_name, data, solution, [])
    elif format == "j30":
        data = load_j30(path)
        solution = load_j30_solution(solution_path, instance_name)

        instance = RCPSP(benchmark_name, instance_name, data, solution, [])
    elif format == "patterson":
        data = load_patterson(path)
        solution = load_patterson_solution(solution_path)

        instance = RCPSP(benchmark_name, instance_name, data, solution, [])
    elif format == "jobshop":
        data = load_jobshop(path)
        solution = load_jobshop_solution(solution_path, instance_name)

        instance = JobShop(benchmark_name, instance_name, data, solution, [])
    
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
    else:
        raise ValueError("Invalid instance kind, should be one of: MMRCPSP, RCPSP, JOBSHOP")

    return instance