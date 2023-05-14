from src.parsers.c15 import load_c15, load_c15_solution
from src.parsers.j30 import load_j30, load_j30_solution
from src.parsers.patterson import load_patterson # , load_patterson_solution
from src.parsers.jobshop import load_jobshop
from docplex.cp.model import CpoModel

from src.mm_rcpsp import MMRCPSP
from src.rcpsp import RCPSP
from src.jobshop import JobShop
from src.optimization_problem import Benchmark
from pathlib import Path


def load_benchmark(directory_path, format):
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)

    benchmark_instances = {}
    for instance in directory_path.iterdir():
        if instance.is_file():
            instance_path = str(instance).replace("\\", "/")
            print(instance)
            print(instance_path)
            instance_name = instance_path.split("/")[-1].split(".")[0]
            instance = load_instance(instance_path, format)

            benchmark_instances[instance_name] = instance
    
    benchmark = Benchmark("test benchmark", benchmark_instances, format)
    return benchmark
        


def load_instance(path, format):
    benchmark_name = path.split("/")[-2]
    instance_name = path.split("/")[-1].split(".")[0]
    
    if format == "c15":
        data = load_c15(path)
        solution = load_c15_solution("data/mm-rcpsp/c15opt.mm.html", instance_name)

        instance = MMRCPSP(benchmark_name, instance_name, format, data, solution, {})
    elif format == "j30":
        data = load_j30(path)
        solution = load_j30_solution("data/mm-rcpsp/c15opt.mm.html", instance_name)

        instance = RCPSP(benchmark_name, instance_name, format, data, solution, {})
    elif format == "patterson":
        raise NotImplementedError("Patterson is not implemented yet")
        return load_patterson(path)
    elif format == "jobshop":
        data = load_jobshop(path)
        # solution = load_jobshop_solution()

        instance = JobShop(benchmark_name, instance_name, format, data, None, {})
    
    return instance