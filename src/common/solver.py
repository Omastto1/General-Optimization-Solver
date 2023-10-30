from docplex.cp.model import CpoParameters
from abc import ABC, abstractmethod

class Solver(ABC):

    @abstractmethod
    def solve(self):
        pass


class CPSolver:
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

    @abstractmethod
    def solve(self):
        """Abstract solve method for CP solver."""
        pass


class GASolver(Solver):
    def __init__(self, seed=None):
        self.seed = seed

    @abstractmethod
    def solve(self):
        """Abstract solve method for GP solver."""
        pass