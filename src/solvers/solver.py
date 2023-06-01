from docplex.cp.model import CpoParameters

class Solver:
    def __init__(self, TimeLimit=60, no_workers=0):
        self.solved = False
        # self.TimeLimit = TimeLimit
        self.params = CpoParameters()
        # params.SearchType = 'Restart'
        # self.params.LogPeriod = 100000
        self.params.LogVerbosity = 'Terse'
        self.params.TimeLimit = TimeLimit

        if no_workers > 0:
            self.params.Workers = no_workers

        print(f"Time limit set to {TimeLimit} seconds" if TimeLimit is not None else "Time limit not restricted")

    def solve_cp(self, instance, validate=False, visualize=False, force_execution=False):
        raise ValueError("CP solver not supported for {instance._instance_kind}.")
    
    def solve_gp(self, instance, validate=False, visualize=False, force_execution=False):
        raise ValueError("GP solver not yet implemented.")
        # raise ValueError("GP solver not supported for {instance._instance_kind}.")
    
    def solve(self, instance, method="CP", validate=False, visualize=False, force_execution=False):
        if method == "CP":
            print("Running CP solver")
            solution, xs = self._solve_cp(instance, validate, visualize, force_execution)
        elif method == "GP":
            print("Running GP solver")
            solution, xs = self._solve_gp(instance, validate, visualize, force_execution)
        else:
            raise Exception("Method not recognized.")
        
        return solution, xs