# Algorithms package initialization
from .greedy import solve_greedy
from .simulated_annealing import solve_simulated_annealing, solve_sa
from .genetic_algorithm import solve_genetic_algorithm
from .nsga2 import solve_nsga2

__all__ = [
    "solve_greedy",
    "solve_simulated_annealing",
    "solve_sa",
    "solve_genetic_algorithm",
    "solve_nsga2",
]
