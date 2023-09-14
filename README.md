# General-Optimization-Solver


DOCS:

# Parsers
Collection of '{benchmark_name}.py' modules which should contain `load_{benchmark_name}` which load instance file and `load_{benchmark_name}_solution` which loads instance solution functions.
instance loader should return json with keys similar to the specific PROBLEM class
instance solution loader should return following json {"feasible": None, "optimum": None, "cpu_time": None, "bounds": None} (cpu_time and bounds being optional)
    
# Solvers
Solver class:
    - __init__(TimeLimit, no_workers):
        self.solved = False
        self.params = CpoParameters()
        self.params.LogVerbosity = 'Terse'
        self.params.TimeLimit = TimeLimit
        self.params.Workers = no_workers

    - solve_cp(instance, validate, visualize, force_execution): - solver specific
    - solve_gp(instance, validate, visualize, force_execution): - solver specific
    
    - solve(instance, method, validate, visualize, force_execution): -- abstract

# PROBLEMS
Benchmark class:
    - __init__(name, instances):
        _name
        _instances

    - solve(solver, method, force_dump)
      - for each instance in benchmark, run solver.solve method 
      - accepts either "CP" or "GP" method

    - dump()
      - calls OptimizationProble.dump_json for each instance in benchmark


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

    - load(path):
    
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

