# System Architecture: Quantum-Rolling-Stock

## Overview

This repository provides a modular, research-grade benchmarking suite for the **Rolling Stock Scheduling and Circulation Problem**. The software architecture decouples data ingestion, objective evaluation, spatial-temporal feasibility verification, optimization algorithms, and visual reporting.

```
quantum-rolling-stock/
├── data/                          # Problem instance data
│   ├── stations.csv               # Network stations and depot designations
│   ├── distance_matrix.csv        # Precomputed pairwise track distances and runtimes
│   ├── fleet.csv                  # Rolling stock fleet specifications and depot stabling
│   ├── trips.csv                  # Timetable scheduled trips
│   ├── demand.csv                 # Passenger demand per trip
│   ├── maintenance_rules.csv      # Type-specific maintenance distance limits
│   └── instance_meta.json         # Instance generator configuration metadata
├── src/
│   ├── utils/                     # Shared core utilities
│   │   ├── __init__.py            # Exported package symbols
│   │   ├── data_loader.py         # load_instance(), get_lookups(), travel()
│   │   ├── cost.py                # compute_cost() multi-criteria evaluation
│   │   ├── feasibility.py         # is_feasible_arc(), is_feasible_chain()
│   │   ├── greedy_init.py         # greedy_construction(), chains_to_schedule_df()
│   │   └── viz.py                 # plot_cost_history(), plot_comparison_bar()
│   ├── algorithms/                # Solver implementations
│   │   ├── __init__.py            # Algorithm solvers export
│   │   ├── simulated_annealing.py # Simulated Annealing with geometric cooling
│   │   ├── greedy.py              # Pure Greedy dispatch baseline
│   │   ├── genetic_algorithm.py   # Single-objective Genetic Algorithm (GA)
│   │   └── nsga2.py               # Multi-objective Evolutionary Algorithm (NSGA-II)
│   └── run_all.py                 # Unified benchmarking script
├── outputs/                       # Algorithm results and figures
│   ├── sa/                        # Simulated Annealing schedules and logs
│   ├── greedy/                    # Greedy baseline outputs
│   ├── ga/                        # Genetic Algorithm outputs
│   ├── nsga2/                     # NSGA-II Pareto front and outputs
│   └── comparison/                # Comparative benchmark CSVs and charts
├── docs/
│   └── math/                      # LaTeX formulation documents
│       ├── problem_formulation.tex
│       ├── simulated_annealing.tex
│       ├── greedy.tex
│       ├── genetic_algorithm.tex
│       └── nsga2.tex
├── architecture.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Component Architecture

### 1. Data Loader & Lookups (`src/utils/data_loader.py`)
- Standardizes ingestion of timetable, fleet, station, demand, and maintenance definitions.
- Converts pandas DataFrames into in-memory dictionary lookups for high-throughput inner loops.

### 2. Objective Function (`src/utils/cost.py`)
- Computes multi-objective penalized cost:
  - Deadhead transit kilometers ($W_{\text{deadhead}} = 1.0$)
  - Activated unit fleet size ($W_{\text{unit}} = 150.0$)
  - Maintenance threshold exceedance ($W_{\text{maint}} = 5.0$)
  - Unmet passenger demand ($W_{\text{demand}} = 2.0$)
  - Infeasible timing transitions ($W_{\text{timing}} = 1000.0$)

### 3. Feasibility Verification (`src/utils/feasibility.py`)
- Enforces train movement continuity and physical turnaround windows.
- Validates chain-level transitions and depot departure times.

### 4. Initialization Heuristic (`src/utils/greedy_init.py`)
- Provides deterministic greedy schedule construction.
- Acts as a shared starting solution for local search and metaheuristics.

### 5. Metaheuristic Solvers (`src/algorithms/`)
- **Greedy Baseline (`greedy.py`)**: Instant deterministic assignment by earliest availability and minimal deadhead.
- **Simulated Annealing (`simulated_annealing.py`)**: Stochastic local search using relocate, swap, and coupling neighborhood moves with geometric cooling.
- **Genetic Algorithm (`genetic_algorithm.py`)**: Population-based evolutionary optimization using tournament selection, uniform trip crossover, and elitism.
- **NSGA-II (`nsga2.py`)**: Multi-objective Pareto optimization balancing operational resource cost against reliability and constraint penalties.

### 6. Benchmark Orchestrator (`src/run_all.py`)
- Runs all 4 algorithms under identical instance configurations.
- Compiles metrics into comparative CSV tables, JSON summaries, and publication-ready charts.
