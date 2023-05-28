class Solver:
    def __init__(self):
        self.solved = False

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