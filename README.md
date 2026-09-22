# Quantum Annealing-Based Rolling Stock Scheduling

A research-grade benchmarking suite and mathematical formulation for the railway rolling stock circulation problem, establishing rigorous classical baselines for QUBO and quantum annealing optimization.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research](https://img.shields.io/badge/Status-Research-orange.svg)](#research-paper-status)

---

## Problem Statement

Rolling stock scheduling and circulation requires assigning physical trainsets to an operational passenger timetable while simultaneously satisfying train coupling limits, turnaround intervals, depot capacities, and periodic maintenance mileage bounds. As railway networks scale, the combinatorial explosion of coupling configurations and spatio-temporal transit dependencies makes finding conflict-free, cost-optimal circulations computationally intractable ($NP$-hard) for classical heuristics. Formulating this combinatorial challenge as a Quadratic Unconstrained Binary Optimization (QUBO) problem maps circulation physics directly to Ising spin systems, allowing quantum annealers to exploit quantum tunneling and superposition to escape deep local minima that trap classical methods.

---

## Algorithms Implemented

| Algorithm | Type | Status |
| :--- | :--- | :---: |
| **Greedy** | Classical | ✅ |
| **Simulated Annealing** | Classical Metaheuristic | ✅ |
| **Genetic Algorithm** | Evolutionary | ✅ |
| **NSGA-II** | Multi-objective Evolutionary | ✅ |
| **Quantum Annealing (QUBO)** | Quantum | 🔜 |

---

## Results Preview

Benchmark performance across a standardized 60-trip, 24-unit timetable instance:

| Algorithm | Best Cost | Units Used | Timing Violations |
| :--- | :---: | :---: | :---: |
| **Greedy** | 5,827.2 | 20 | 0 |
| **Simulated Annealing** | 4,024.4 | 19 | 0 |
| **Genetic Algorithm** | 10,203.1 | 24 | 2 |
| **NSGA-II** | 9,363.9 | 19 | 4 |
| **Quantum Annealing (QUBO)** | *In Progress* | *TBD* | *TBD* |

> [!NOTE]
> Detailed metrics, convergence trajectories, and Pareto front trade-offs are logged under `outputs/comparison/master_comparison.csv` and documented in [architecture.md](architecture.md).

---

## Folder Structure

```
quantum-rolling-stock/
├── data/
│   ├── stations.csv               # Railway stations, depots, and maintenance facility designations
│   ├── distance_matrix.csv        # Precomputed pairwise station travel times and track distances
│   ├── fleet.csv                  # Rolling stock trainsets, capacities, home depots, and initial mileage
│   ├── trips.csv                  # Daily timetable trips with origins, destinations, and timings
│   ├── demand.csv                 # Expected peak passenger demand per timetable trip
│   ├── maintenance_rules.csv      # Maximum permissible distance intervals between overhauls per unit type
│   └── instance_meta.json         # Metadata and configuration settings for the problem instance
├── src/
│   ├── utils/
│   │   ├── __init__.py            # Export package interface for shared utility functions
│   │   ├── data_loader.py         # Standardized instance loader and fast dictionary index builder
│   │   ├── cost.py                # Multi-criteria objective cost evaluation and penalty breakdown
│   │   ├── feasibility.py         # Temporal and spatial arc/chain feasibility verification routines
│   │   ├── greedy_init.py         # Deterministic greedy construction heuristic and schedule DataFrame exporter
│   │   └── viz.py                 # Convergence trajectory, bar chart, and failure heatmap plotting
│   ├── algorithms/
│   │   ├── __init__.py            # Package interface exposing all optimization solver entrypoints
│   │   ├── greedy.py              # Pure greedy dispatch baseline solver with diagnostic failure analysis
│   │   ├── simulated_annealing.py # Metropolis simulated annealing metaheuristic with geometric cooling
│   │   ├── genetic_algorithm.py   # Single-objective evolutionary algorithm with PMX crossover and elitism
│   │   └── nsga2.py               # Non-dominated sorting genetic algorithm II for Pareto front optimization
│   ├── run_all.py                 # Unified benchmark orchestrator executing all 4 algorithms sequentially
│   ├── run_failure_analysis.py    # Classical failure diagnostic comparison and chart generation script
│   └── scaling_failure_simulation.py # Computational complexity and constraint breakdown scaling simulator
├── outputs/
│   ├── greedy/
│   │   ├── .gitkeep               # Directory placeholder tracking greedy outputs folder in Git
│   │   ├── final_schedule.csv     # Unit-by-unit timetable trip circulation plan from greedy dispatch
│   │   ├── run_summary.json       # Cost breakdown and operational metrics for greedy baseline
│   │   ├── cost_history.png       # Flat baseline cost trajectory visualization
│   │   └── failure_analysis.json  # Diagnostic log of timing, uncovered, and overloaded trips
│   ├── sa/
│   │   ├── .gitkeep               # Directory placeholder tracking simulated annealing outputs
│   │   ├── final_schedule.csv     # Unit-by-unit circulation schedule produced by simulated annealing
│   │   ├── run_summary.json       # Initial vs final cost breakdown and percentage improvement
│   │   ├── sa_cost_history.csv    # Iteration-by-iteration temperature and cost progression log
│   │   └── cost_history.png       # Convergence curve of best and current costs over cooling steps
│   ├── ga/
│   │   ├── .gitkeep               # Directory placeholder tracking genetic algorithm outputs
│   │   ├── final_schedule.csv     # Unit circulation schedule corresponding to the best chromosome
│   │   ├── run_summary.json       # Hyperparameters, completed generations, and final cost breakdown
│   │   ├── ga_cost_history.csv    # Generational log of best and average population costs
│   │   ├── cost_history.png       # Evolutionary cost convergence trajectory plot
│   │   └── failure_analysis.json  # Log of generations with timing violations and infeasible trips
│   ├── nsga2/
│   │   ├── .gitkeep               # Directory placeholder tracking NSGA-II outputs
│   │   ├── final_schedule.csv     # Best compromise circulation plan selected from Pareto front
│   │   ├── pareto_front.csv       # Non-dominated solution set across the three objective dimensions
│   │   ├── run_summary.json       # Pareto front size and min/max statistics per objective
│   │   ├── cost_history.csv       # Generational log of best f1, f2, f3, and Pareto set cardinality
│   │   ├── cost_history.png       # Convergence trajectory of primary objective f1 (deadhead km)
│   │   └── pareto_front.png       # 3D scatter visualization of the Pareto front colored by crowding distance
│   └── comparison/
│       ├── .gitkeep               # Directory placeholder tracking comparison folder in Git
│       ├── master_comparison.csv  # Side-by-side benchmark table comparing all 4 solvers across all metrics
│       ├── master_comparison.png  # 4-panel bar chart comparing total cost, deadhead, units, and demand
│       ├── failure_summary.json   # Direct numerical comparison of failure metrics between Greedy and SA
│       ├── classical_failure_chart.png # Grouped bar chart illustrating feasibility threshold breaches
│       ├── cost_reduction_chart.png # SA cost reduction trajectory annotated against greedy and GA reference
│       ├── scaling_failure.csv    # Scaling experiment table recording violations and runtimes vs n_trips
│       └── scaling_failure.png    # Dual-axis line plot demonstrating greedy breakdown at network scale
├── docs/
│   └── math/
│       ├── problem_formulation.tex # Formal LaTeX mathematical model of the circulation problem
│       ├── simulated_annealing.tex # LaTeX documentation of state representation, moves, and cooling
│       ├── greedy.tex             # LaTeX specification of sequential greedy dispatch and state logic
│       ├── genetic_algorithm.tex  # LaTeX formulation of chromosome encoding, crossover, and elitism
│       ├── nsga2.tex              # LaTeX specification of non-dominated sorting and crowding distance
│       └── rolling_stock_mathematical_formulation.pdf # Original reference PDF formulation
├── generate_data.py               # Synthetic instance generator creating stations, trips, fleet, and demand
├── architecture.md                # System architectural specification and technical documentation
├── README.md                      # High-level project overview, installation, and user instructions
├── requirements.txt               # Pinned Python package dependencies for reproduction
└── .gitignore                     # Git ignore rules for bytecode, cache, and generated PNG figures
```

---

## Quickstart

Run the entire pipeline in three commands:

```bash
pip install -r requirements.txt
python generate_data.py
python src/run_all.py
```

All benchmark schedules, JSON performance summaries, and comparative visualizations are automatically generated in `outputs/`.

---

## Research Paper Status

- **Status**: In Progress
- **Target Venue**: XXXXXXX conference
- **Working Title**: *Benchmarking Classical and Quantum Annealing Approaches for Rolling Stock Scheduling Under Complex Coupling and Maintenance Constraints*

---

## Contributing / Citation

Contributions, issue reports, and experimental extensions are welcome. Please open an issue or pull request following the repository coding standards.

If you utilize this benchmarking framework or mathematical formulation in your research, please cite:

```bibtex
@misc{quantum_rolling_stock_2026,
  author       = {Mandal, Sayantan and Contributors},
  title        = {Quantum Annealing-Based Rolling Stock Scheduling Benchmarks},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/Thunder7is/Quantum}}
}
```