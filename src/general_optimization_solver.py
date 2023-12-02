import os
import json
from pathlib import Path

from src.mmrcpsp.c15 import load_c15, load_c15_solution
from src.rcpsp.j30 import load_j30, load_j30_solution, load_j30_ugent_csv_solution
from src.rcpsp.patterson import load_patterson, load_patterson_solution
from src.jobshop.parser import load_jobshop, load_jobshop_solution
from src.strippacking2d.parser import load_strip_packing, load_strip_packing_solution, load_bkw_benchmark, load_bkw_benchmark_solution
from src.mmrcpsp.mmlib import load_mmlib, load_mmlib_solution
from src.binpacking1d.parser import load_1dbinpacking
from src.binpacking2d.parser_no_items_first import load_2dbinpacking_no_items_first
from src.binpacking2d.parser_bin_size_first import load_2dbinpacking_bin_size_first  # TODO

from src.common.optimization_problem import Benchmark
from src.mmrcpsp.problem import MMRCPSP
from src.rcpsp.problem import RCPSP
from src.jobshop.problem import JobShop
from src.strippacking2d.problem import StripPacking2D
from src.binpacking1d.problem import BinPacking1D
from src.binpacking2d.problem import BinPacking2D


def load_raw_benchmark(directory_path, solution_path=None, format=None, no_instances=0, force_dump=True):
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)

    if format is None:
        meta_path = Path(directory_path) / '.meta'
        meta_data = _get_and_validate_meta_data(meta_path)

        format = meta_data["FORMAT"]
        if solution_path is None:
            solution_path = meta_data.get("SOLUTION_PATH", None)
            print("Loading .meta solution path: %s" % solution_path)

    if solution_path is None:
        print("\nWARNING: Loading benchmark instances without solutions\n")

    print("Loading raw benchmark data")
    benchmark_instances = {}
    instances = [file for file in directory_path.iterdir()
                 if file.name != '.meta']
    for i, instance in enumerate(instances):
        if no_instances > 0 and i >= no_instances:
            break

        if instance.is_file() and instance.name != ".meta":
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
    for i, instance in enumerate(directory_path.iterdir()):
        if no_instances > 0 and i >= no_instances:
            break

        if instance.is_file():
            instance_path = str(instance).replace("\\", "/")
            instance_name = instance_path.split("/")[-1].split(".")[0]
            instance = load_instance(instance_path)

            benchmark_instances[instance_name] = instance

    benchmark = Benchmark("test benchmark", benchmark_instances)
    return benchmark


def _get_and_validate_meta_data(filepath):
    """
    Reads and validates a .meta file, returning a dictionary of metadata.

    The function checks if the .meta file exists, parses its content, and validates
    that each line conforms to the expected key-value format. It ensures that each key
    is one of the predefined valid keys. If the file is valid, it returns a dictionary
    containing the metadata. If the file is invalid or missing, it raises an error.

    Parameters:
    filepath (str): The path to the .meta file.

    Returns:
    dict: A dictionary containing the metadata from the .meta file.

    Raises:
    FileNotFoundError: If the .meta file does not exist.
    ValueError: If the file contains invalid or unexpected content.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Metadata file {filepath} does not exist.\nPlease create .meta file or provide `format` parameter.")

    valid_keys = {"FORMAT", "SOLUTION_PATH"}  # Add more valid keys as needed
    metadata = {}

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('=')
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid line in metadata file: {line.strip()}")

            key, value = parts
            key = key.strip().upper()  # Normalize the key
            if key not in valid_keys:
                raise ValueError(f"Unexpected key in metadata file: {key}")

            metadata[key] = value.strip()

    return metadata


def load_raw_instance(path, solution_path, format=None, verbose=False):
    benchmark_name = path.split("/")[-2].split(".")[0]
    instance_name = path.split("/")[-1].split(".")[0]

    if format is None:
        meta_path = Path(path).parent / '.meta'
        meta_data = _get_and_validate_meta_data(meta_path)

        format = meta_data["FORMAT"]

    assert format is not None, "Specify valid raw data input argument `format`"

    # TODO: ADD SOLUTION PATH TO METADATA
    if format == "c15":
        data = load_c15(path, verbose)
        if solution_path:
            solution = load_c15_solution(
                solution_path, benchmark_name, instance_name)
        else:
            solution = {}

        instance = MMRCPSP(benchmark_name, instance_name, data, solution, [])
    elif format == "j30":
        data = load_j30(path, verbose)
        if solution_path:
            if solution_path.endswith(".csv"):
                solution = load_j30_ugent_csv_solution(solution_path, benchmark_name, instance_name)
            else:
                solution = load_j30_solution(
                    solution_path, benchmark_name, instance_name)
        else:
            solution = {}

        instance = RCPSP(benchmark_name, instance_name, data, solution, [])
    elif format == "patterson":
        print(solution_path, instance_name)
        data = load_patterson(path, verbose)
        if solution_path:
            solution = load_patterson_solution(solution_path, instance_name)
        else:
            solution = {}

        instance = RCPSP(benchmark_name, instance_name, data, solution, [])
    elif format == "jobshop":
        data = load_jobshop(path, verbose)
        if solution_path:
            solution = load_jobshop_solution(solution_path, instance_name)
        else:
            solution = {}

        instance = JobShop(benchmark_name, instance_name, data, solution, [])
    elif format == "strippacking":
        data = load_strip_packing(path, verbose)
        if solution_path:
            solution = load_strip_packing_solution(
                solution_path, instance_name)
        else:
            solution = {}

        instance = StripPacking2D(
            benchmark_name, instance_name, data, solution, [])
    elif format == 'bkw':
        data = load_bkw_benchmark(path, verbose)
        if solution_path:
            solution = load_bkw_benchmark_solution(solution_path, instance_name)
        else:
            solution = {}

        instance = StripPacking2D(benchmark_name, instance_name, data, solution, [])

    elif format == "1Dbinpacking":
        data = load_1dbinpacking(path, verbose)
        solution = {}
        # TODO: SO FAR USINGBENCHMARK WITH NO SOLUTION
        # solution = load_strip_packing_solution(solution_path, instance_name)

        instance = BinPacking1D(
            benchmark_name, instance_name, data, solution, [])
    elif format == "2Dbinpacking":
        data = load_2dbinpacking_no_items_first(path, verbose)
        solution = {}
        # TODO: SO FAR USINGBENCHMARK WITH NO SOLUTION
        # solution = load_strip_packing_solution(solution_path, instance_name)

        instance = BinPacking2D(
            benchmark_name, instance_name, data, solution, [])
    elif format == "2Dbinpacking_bin_size_first":
        data = load_2dbinpacking_bin_size_first()(path, verbose)
        solution = {}
        # TODO: SO FAR USINGBENCHMARK WITH NO SOLUTION
        # solution = load_strip_packing_solution(solution_path, instance_name)

        instance = BinPacking2D(
            benchmark_name, instance_name, data, solution, [])
    elif format == "mmlib":
        data = load_mmlib(path, verbose)
        if solution_path:
            solution = load_mmlib_solution(solution_path, instance_name)
        else:
            solution = {}

        instance = MMRCPSP(benchmark_name, instance_name, data, solution, [])
    else:
        raise ValueError(
            "Invalid format, should be one of: c15, j30, patterson, jobshop, strippacking, 1Dbinpacking, 2Dbinpacking, mmlib")

    return instance


def load_instance(path):
    assert path.endswith(
        ".json"), "Invalid input file, should be .json with predefined structure"

    with open(path, "r") as f:
        instance_dict = json.load(f)

    benchmark_name = instance_dict["benchmark_name"]
    instance_name = instance_dict["instance_name"]
    instance_kind = instance_dict["instance_kind"]
    data = instance_dict["data"]
    solution = instance_dict["reference_solution"]
    run_history = instance_dict["run_history"]

    if instance_kind == "MMRCPSP":
        instance = MMRCPSP(benchmark_name, instance_name,
                           data, solution, run_history)
    elif instance_kind == "RCPSP":
        instance = RCPSP(benchmark_name, instance_name,
                         data, solution, run_history)
    elif instance_kind == "JOBSHOP":
        instance = JobShop(benchmark_name, instance_name,
                           data, solution, run_history)
    elif instance_kind == "2DSTRIPPACKING":
        instance = StripPacking2D(
            benchmark_name, instance_name, data, solution, run_history)
    elif instance_kind == "1DBINPACKING":
        instance = BinPacking1D(
            benchmark_name, instance_name, data, solution, run_history)
    else:
        raise ValueError(
            "Invalid instance kind, should be one of: MMRCPSP, RCPSP, JOBSHOP")

    return instance
