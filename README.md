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

| Algorithm | Best Cost | Deadhead km | Units Used | Uncovered Demand | Timing Violations |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Greedy** | 5,696.9 | 1,184.9 | 20 | 756 | 0 |
| **Simulated Annealing** | 4,074.1 | 966.1 | 19 | 129 | 0 |
| **Genetic Algorithm** | 7,910.6 | 2,574.6 | 23 | 943 | 0 |
| **NSGA-II** | 7,807.1 | 2,205.1 | 24 | 501 | 1 |
| **Quantum Annealing (QUBO)** | *In Progress* | *TBD* | *TBD* | *TBD* | *TBD* |

> [!NOTE]
> Detailed metrics, convergence trajectories, and Pareto front trade-offs are logged under `outputs/comparison/master_comparison.csv` and documented in [architecture.md](architecture.md).

---

## Folder Structure

```
quantum-rolling-stock/
├── data/
│   ├── stations.csv               # station nodes and depot flags
│   ├── distance_matrix.csv        # pairwise travel time and km
│   ├── fleet.csv                  # trainset types and initial state
│   ├── trips.csv                  # daily timetable with timings
│   ├── demand.csv                 # passenger demand per trip
│   ├── maintenance_rules.csv      # max km per unit type
│   └── instance_meta.json         # instance config metadata
├── src/
│   ├── utils/
│   │   ├── data_loader.py         # loads and indexes instance data
│   │   ├── cost.py                # shared objective cost function
│   │   ├── feasibility.py         # arc and chain feasibility checks
│   │   ├── greedy_init.py         # greedy construction heuristic
│   │   └── viz.py                 # shared plotting utilities
│   ├── algorithms/
│   │   ├── greedy.py              # pure greedy baseline solver
│   │   ├── simulated_annealing.py # SA with geometric cooling
│   │   ├── genetic_algorithm.py   # GA with PMX crossover
│   │   └── nsga2.py               # multi-objective Pareto solver
│   ├── run_all.py                 # runs all 4 solvers sequentially
│   ├── run_failure_analysis.py    # classical failure diagnostics
│   └── scaling_failure_simulation.py # all-algo scaling experiment
├── outputs/
│   ├── greedy/                    # greedy schedule and metrics
│   ├── sa/                        # SA schedule, cost history
│   ├── ga/                        # GA schedule, generations log
│   ├── nsga2/                     # Pareto front and 3D plot
│   └── comparison/                # master comparison charts
├── docs/math/                     # LaTeX formulations per algorithm
├── generate_data.py               # synthetic instance generator
├── architecture.md                # full system architecture doc
└── requirements.txt               # Python dependencies
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

## Contributors

| Name | Role |
| :--- | :--- |
| Sayantan Mandal | Algorithm design, MLOps, project lead |
| Rishav Pal | Data generation, experimental setup |
| Asmit Sharma | Mathematical formulation, LaTeX |
| Ashish Kumar | Visualization, analysis, documentation |

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
