from abc import ABC, abstractmethod


class Solver(ABC):

    @abstractmethod
    def read_json(self, fname):
        pass

    @abstractmethod
    def save_to_json(self):
        pass

    @abstractmethod
    def solve(self, tlim):
        pass

    @abstractmethod
    def visualize_solution(self):
        pass

    @abstractmethod
    def validate_solution(self):
        pass
