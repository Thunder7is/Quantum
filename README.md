# Quantum Annealing-Based Rolling Stock Scheduling

A research-grade benchmarking suite for the **Rolling Stock Scheduling and Circulation Problem**, comparing classical heuristics, metaheuristics, multi-objective evolutionary algorithms, and preparing the foundation for QUBO / Quantum Annealing formulations.

---

## Repository Layout

```
quantum-rolling-stock/
├── data/                          # Raw instance data
│   ├── stations.csv               # Station network topology
│   ├── distance_matrix.csv        # Travel times and track distances
│   ├── fleet.csv                  # Rolling stock trainsets and capacities
│   ├── trips.csv                  # Scheduled passenger trips
│   ├── demand.csv                 # Expected passenger demand per trip
│   ├── maintenance_rules.csv      # Maintenance distance limits
│   └── instance_meta.json         # Generator configuration
├── src/
│   ├── utils/                     # Shared modular utilities
│   │   ├── __init__.py
│   │   ├── data_loader.py         # load_instance(), get_lookups()
│   │   ├── cost.py                # compute_cost() shared objective
│   │   ├── feasibility.py         # is_feasible_arc(), is_feasible_chain()
│   │   ├── greedy_init.py         # greedy_construction() shared starting point
│   │   └── viz.py                 # plot_cost_history(), plot_comparison_bar()
│   ├── algorithms/                # Modular solvers
│   │   ├── __init__.py
│   │   ├── simulated_annealing.py # Refactored SA solver
│   │   ├── greedy.py              # Pure greedy baseline
│   │   ├── genetic_algorithm.py   # Genetic Algorithm (GA)
│   │   └── nsga2.py               # NSGA-II multi-objective solver
│   └── run_all.py                 # Unified benchmark execution script
├── outputs/                       # Algorithm results and figures
│   ├── sa/                        # SA final schedule, cost history, plots
│   ├── greedy/                    # Greedy baseline outputs
│   ├── ga/                        # GA schedule and convergence logs
│   ├── nsga2/                     # NSGA-II Pareto front and plots
│   └── comparison/                # Comparative benchmark CSVs and charts
├── docs/
│   └── math/                      # LaTeX mathematical formulations
│       ├── problem_formulation.tex
│       ├── simulated_annealing.tex
│       ├── greedy.tex
│       ├── genetic_algorithm.tex
│       └── nsga2.tex
├── architecture.md                # System architectural design
├── README.md                      # Project documentation
├── requirements.txt               # Python package dependencies
└── .gitignore                     # Git ignore rules
```

---

## Installation

Ensure Python 3.10+ is installed:

```bash
pip install -r requirements.txt
```

---

## Running Solvers

### 1. Run Complete Benchmark (All 4 Algorithms)
To execute all algorithms, record runtimes, compare performance, and generate comparison charts:

```bash
python3 src/run_all.py
```

Results are saved to `outputs/comparison/algorithm_comparison.csv` and `outputs/comparison/comparison_bar.png`.

### 2. Run Individual Solvers
Each algorithm can be executed independently:

```bash
# Pure Greedy Baseline
python3 -m src.algorithms.greedy

# Simulated Annealing
python3 -m src.algorithms.simulated_annealing

# Genetic Algorithm
python3 -m src.algorithms.genetic_algorithm

# NSGA-II Multi-Objective Optimization
python3 -m src.algorithms.nsga2
```

---

## Optimization Objectives

The objective minimizes a penalized operational cost function:
- **Deadhead Transit**: Repositioning empty trainsets between stations ($1.0 / \text{km}$).
- **Unit Activation**: Fixed operational cost per activated trainset ($150.0 / \text{unit}$).
- **Maintenance Overrun**: Penalty for exceeding maintenance intervals ($5.0 / \text{km}$).
- **Unmet Demand**: Passenger capacity shortfall on scheduled trips ($2.0 / \text{passenger}$).
- **Timing Infeasibility**: Strict penalty for unfeasible transit transitions ($1000.0 / \text{occurrence}$).

Mathematical models are documented in `docs/math/`.