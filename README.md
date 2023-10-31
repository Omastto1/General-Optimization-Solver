# General-Optimization-Solver


## CLASS DOCS:

## PROJECT STRUCTURE
Each problem type has its own directory, which contains `solver` directory with solver modules and `problem.py` module as well as `parser` modules, if only one input format is implemented or multiple modules with respective format names (`c15.py` and `mmlib.py` for MM-RCPSP)

Base classes are located in `common` directory

All important classes are imported to the `general_optimization_solver.py` module, which is the main entry point for the project

### Framework API

  - load_raw_benchmark(directory_path, solution_path, format, no_instances, force_dump)
    - load raw data from specified files in a specified format

  - load_benchmark(directory_path, no_instances)
    - load unified data from json files in specified directory

  - load_raw_instance(path, solution_path, format, verbose)

  - load_instance(path):

### Parsers

Collection of '{benchmark_name}.py' modules which should contain `load_{benchmark_name}` which load instance file and `load_{benchmark_name}_solution` which loads instance solution functions.  

Instance loader should return json with keys similar to the specific PROBLEM class.  

Instance solution loader should return following json {"feasible": None, "optimum": None, "cpu_time": None, "bounds": None} (cpu_time and bounds being optional)
    
### Solvers

Solver class:
    
  - solve(): -- abstract

CPSolver class:  

  - __init__(TimeLimit, no_workers):  
      - solved  
      - params  

  - solve(instance, validate, visualize, force_execution): - CP solver specific (base abstract)
    - so far returns docplex solution and cp variables

  - add_run_to_history(instance, sol): - CP Solver specific (implemented in base class)
    - extracts common info from docplex solution and updates the instance run history
  
GASolver class:

  - solve(algorithm, instance, fitness_func, termination, validate, visualize, force_execution): - GA solver specific
    - returns objective value, {other important solution values like start times for rcpsp}, pymoo solution object

  - add_run_to_history(instance, objective_value, start_times, is_valid): - GA Solver specific (implemented in base class)
    - extracts common info from pymoo solution and updates the instance run history

### PROBLEMS
Benchmark class:  

    - __init__(name, instances):  
        _name  
        _instances  

    - solve(solver, method, solver_config, force_dump)
      - for each instance in benchmark, run solver.solve method 
      - solver config is method (CP/GA) specific

    - dump()
      - calls OptimizationProble.dump_json for each instance in benchmark

    - generate_solver_comparison_markdown(instances_subset, methods_subset):
      - generate markdown table with comparison of solver results for given instances and methods


OptimizationProblem class

  - __init__(benchmark_name, instance_name, _instance_kind, data, solution, run_history):  
    - General config  
      - _benchmark_name  
      - _instance_name  
      - _instance_kind  
      - _data  
      - _solution  
      - _run_history  
    - Problem specific config  
      - ...
  
  - dump():
    - save json to "data/{self._instance_kind}/{self._benchmark_name}/{self._instance_name}.json"
    - dict prescription:
      {
          "benchmark_name": self._benchmark_name,
          "instance_name": self._instance_name,
          "instance_kind": self._instance_kind,

          "data": self._data,
          "reference_solution": self._solution,
          "run_history": self._run_history,
      }
  - compare_to_reference(obj_value):
    - print out whether solution is optimal or worse than optimal if optimum is known
    - print out whether solution is worse than upper / lower bound
    - print out if no solution exists

  - update_run_history(solution, method, solver_params)
  - reset_run_history()

  - skip_on_optimal_solution

  - visualize(solution, variables) - problem specific
  - validate(solution, variables) - problem specific

## TODO LIST
### Tomas
jobshop - MVP - solving, validating constraint, visualization
rcpsp - MVP - solving, validating constraint, visualization
mm-rcpsp - MVP - solving, validating constraint, visualization

validation of solution

Proper name for benchmarks  
Different naming for formats  
add instance type (rcp/mmrcsp/jobshop) to class
save benchmark as json  
Save solutions  
Add solve parameters (Time Limit, etc.)

binpacking - did not find cp model

### Michal

