from general_optimization_solver import *
from src.vrp.solvers.integer_model import VRPTWSolver

solvers_config = {"TimeLimit": 30}
path = "..\\data\\VRPTW\\solomon_25"
benchmark = load_benchmark(path)
solver = VRPTWSolver(**solvers_config)
solver.solve(benchmark, validate=True)
benchmark.dump(dir_path="..\\data\\VRPTW\\solomon_25")

solvers_config = {"TimeLimit": 100}
path = "..\\data\\VRPTW\\solomon_50"
benchmark = load_benchmark(path)
solver = VRPTWSolver(**solvers_config)
solver.solve(benchmark, validate=True)
benchmark.dump(dir_path="..\\data\\VRPTW\\solomon_50")

solvers_config = {"TimeLimit": 300}
path = "..\\data\\VRPTW\\solomon_100"
benchmark = load_benchmark(path)
solver = VRPTWSolver(**solvers_config)
solver.solve(benchmark, validate=True)
benchmark.dump(dir_path="..\\data\\VRPTW\\solomon_100")

