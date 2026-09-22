# System Architecture: Quantum-Rolling-Stock

## 1. Project Overview

The **Rolling Stock Scheduling and Circulation Problem** addresses the operational challenge of assigning rolling stock trainsets to scheduled passenger timetable trips while satisfying physical train coupling limits, maintenance mileage intervals, and spatio-temporal turnaround feasibility. This research-grade benchmarking suite evaluates classical constructive heuristics, simulated annealing metaheuristics, single-objective genetic algorithms, and multi-objective evolutionary optimization (NSGA-II) against standardized operational and penalty metrics, establishing a rigorous experimental baseline for future QUBO formulations on quantum annealers.

---

## 2. Folder Structure

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

## 3. Data Flow Diagram

```
+-----------------------------------------------------------------------------------+
|                                     DATA TIER                                     |
|  data/stations.csv, distance_matrix.csv, fleet.csv, trips.csv, demand.csv, etc.   |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                                 SHARED UTILS TIER                                 |
|  src/utils/data_loader.py: load_instance() -> dict                                |
|    - Ingests timetable, fleet, network geometry, demand, and maintenance rules    |
|    - Generates fast in-memory lookups: dist_lookup, maint_lookup, fleet/trips     |
|  src/utils/feasibility.py: is_feasible_arc(), is_feasible_chain()                 |
|  src/utils/cost.py: compute_cost() -> dict (deadhead, units, maint, demand, time) |
|  src/utils/greedy_init.py: greedy_construction() -> chains dict                   |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                                 ALGORITHMS TIER                                   |
|  ┌───────────────────┐ ┌──────────────────────┐ ┌──────────────────────────────┐  |
|  │  src/algorithms/  │ │   src/algorithms/    │ │       src/algorithms/        │  |
|  │     greedy.py     │ │simulated_annealing.py│ │    genetic_algorithm.py      │  |
|  │ (Pure Baseline 2) │ │ (Metropolis Cooling) │ │(PMX, Tourn=5, 10x Penalty,   │  |
|  │                   │ │                      │ │ 80 Chromosomes, Elitism=2)   │  |
|  └─────────┬─────────┘ └──────────┬───────────┘ └──────────────┬───────────────┘  |
|            │                      │                            │                  |
|            │           ┌──────────┴───────────┐                │                  |
|            │           │   src/algorithms/    │                │                  |
|            │           │       nsga2.py       │                │                  |
|            │           │  (3-Obj Pareto MOEA: │                │                  |
|            │           │   f1:km, f2:u, f3:pen│                │                  |
|            │           │   SBX, Poly Mutation)│                │                  |
|            │           └──────────┬───────────┘                │                  |
|            │                      │                            │                  |
|            └──────────────────────┼────────────────────────────┘                  |
|                                   ▼                                               |
|                    src/run_all.py & run_failure_analysis.py                       |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                                   OUTPUTS TIER                                    |
|  outputs/greedy/       outputs/sa/         outputs/ga/         outputs/nsga2/     |
|  - final_schedule.csv  - final_schedule.csv- final_schedule.csv- final_schedule.csv |
|  - run_summary.json    - run_summary.json  - run_summary.json  - pareto_front.csv |
|  - failure_analysis    - sa_cost_history   - ga_cost_history   - cost_history.csv |
|  - cost_history.png    - cost_history.png  - cost_history.png  - pareto_front.png |
|                                                                - run_summary.json |
|                                                                - cost_history.png |
|                                                                                   |
|  outputs/comparison/                                                              |
|  - master_comparison.csv, master_comparison.png                                   |
|  - failure_summary.json, classical_failure_chart.png, cost_reduction_chart.png    |
|  - scaling_failure.csv, scaling_failure.png                                       |
+-----------------------------------------------------------------------------------+
```

---

## 4. Algorithm Comparison Table

| Algorithm | Type | Objectives | Representation | Key Params |
| :--- | :--- | :--- | :--- | :--- |
| **Greedy** | Classical Constructive Dispatch Heuristic | Single scalarized cost ($W_{\text{deadhead}}=1, W_{\text{unit}}=150, W_{\text{maint}}=5, W_{\text{demand}}=2, W_{\text{timing}}=1000$) | Unit circulation chains: `{unit_id: [trip_id, ...]}` sorted by departure time | Sequential departure dispatch, nearest available unit selection, demand-driven coupling |
| **Simulated Annealing (SA)** | Stochastic Local Search Metaheuristic | Single scalarized cost ($W_{\text{deadhead}}=1, W_{\text{unit}}=150, W_{\text{maint}}=5, W_{\text{demand}}=2, W_{\text{timing}}=1000$) | Unit circulation chains: `{unit_id: [trip_id, ...]}` | $T_0 = 500.0, \alpha = 0.995, T_{\min} = 0.5$, 40 iters/temp, moves: relocate (with arc feasibility check), swap, couple/decouple |
| **Genetic Algorithm (GA)** | Population-Based Evolutionary Algorithm | Single scalarized cost with $10\times$ penalty multiplier on timing violations | Discrete integer chromosome of 60 unit indices ($0 \dots 23$) | Population: 80, Tournament $k=5$, PMX Crossover ($p_c=0.85$), Uniform mutation ($p_m=0.1/\text{gene}$), Elitism: 2, Max Gen: 300, Plateau patience: 50 |
| **NSGA-II** | Multi-Objective Evolutionary Algorithm (MOEA) | Simultaneous 3-objective Pareto minimization: $f_1 = \text{deadhead\_km}$, $f_2 = \text{units\_used}$, $f_3 = \text{uncovered} + 100 \times \text{timing}$ | Discrete integer chromosome of 60 unit indices ($0 \dots 23$) | Population: 60, Generations: 200, Binary tournament on $(\text{rank}, -cd)$, SBX ($\eta=15, p_c=0.9$), Polynomial mutation ($\eta=20, p_m=1/60$), $(\mu + \lambda)$ elitist truncation |

---

## 5. Shared Utils Contract

The [`src/utils/`](file:///Users/sayantanmandal/Downloads/Quantum-main/src/utils) package encapsulates all shared business logic, data models, objective evaluations, and visual plotting utilities:

### `load_instance(data_dir: str = DEFAULT_DATA_DIR) -> dict`
- **Location**: [`src/utils/data_loader.py`](file:///Users/sayantanmandal/Downloads/Quantum-main/src/utils/data_loader.py)
- **Inputs**: `data_dir` (`str`) — Path to instance CSV files.
- **Outputs**: `dict` containing:
  - `"stations"`: `pd.DataFrame` of station nodes and depot facilities.
  - `"fleet"`: `pd.DataFrame` of rolling stock fleet units, home depots, and capacities.
  - `"trips"`: `pd.DataFrame` of timetable trips merged with passenger demand, sorted by `departure_min`.
  - `"dist_lookup"`: `dict` mapping `(from_station, to_station) -> (travel_time_min, distance_km)`.
  - `"maint_lookup"`: `dict` mapping `unit_type -> max_km_between_maintenance`.
  - `"depot_set"`: `set` of station IDs that are depots.
  - `"maint_depot_set"`: `set` of station IDs equipped with maintenance facilities.

### `get_lookups(fleet: pd.DataFrame, trips: pd.DataFrame) -> tuple[dict, dict]`
- **Location**: [`src/utils/data_loader.py`](file:///Users/sayantanmandal/Downloads/Quantum-main/src/utils/data_loader.py)
- **Inputs**: `fleet` (`pd.DataFrame`), `trips` (`pd.DataFrame`).
- **Outputs**: `(fleet_lookup, trips_lookup)` — Pre-indexed dictionaries mapping IDs to record dicts for high-throughput $O(1)$ evaluation loops.

### `travel(dist_lookup: dict, origin: str, destination: str) -> tuple[int, float]`
- **Location**: [`src/utils/data_loader.py`](file:///Users/sayantanmandal/Downloads/Quantum-main/src/utils/data_loader.py)
- **Inputs**: `dist_lookup` (`dict`), `origin` (`str`), `destination` (`str`).
- **Outputs**: `(travel_time_min, distance_km)` — Returns `(0, 0.0)` if origin equals destination.

### `compute_cost(chains: dict, trips_df, fleet_df, dist_lookup: dict, maint_lookup: dict, weights: dict = None) -> dict`
- **Location**: [`src/utils/cost.py`](file:///Users/sayantanmandal/Downloads/Quantum-main/src/utils/cost.py)
- **Inputs**: `chains` (`dict` `{unit_id: [trip_id, ...]}`), `trips_df` (`pd.DataFrame` or `dict`), `fleet_df` (`pd.DataFrame` or `dict`), `dist_lookup` (`dict`), `maint_lookup` (`dict`), optional `weights` (`dict` with keys: `deadhead, activation, maintenance, demand, timing_violation`).
- **Outputs**: `dict` containing:
  - `"total"`: Cumulative scalarized cost (`float`).
  - `"deadhead_km"`: Total empty repositioning distance (`float`).
  - `"units_used"`: Number of activated trainsets (`int`).
  - `"maintenance_violation_km"`: Distance run past maintenance threshold (`float`).
  - `"uncovered_demand"`: Unmet passenger capacity shortfall (`int`).
  - `"timing_violations"`: Number of infeasible transit/departure connections (`int`).

### `is_feasible_arc(unit_row, trip_a_row, trip_b_row, dist_lookup: dict) -> bool`
- **Location**: [`src/utils/feasibility.py`](file:///Users/sayantanmandal/Downloads/Quantum-main/src/utils/feasibility.py)
- **Inputs**: `unit_row` (`dict` or namedtuple), `trip_a_row` (`dict` or namedtuple), `trip_b_row` (`dict` or namedtuple), `dist_lookup` (`dict`).
- **Outputs**: `bool` — `True` if unit can complete trip A, observe minimum turnaround, deadhead to trip B origin, and arrive before trip B scheduled departure.

### `is_feasible_chain(unit_row, trip_ids: list, trips_df, dist_lookup: dict) -> bool`
- **Location**: [`src/utils/feasibility.py`](file:///Users/sayantanmandal/Downloads/Quantum-main/src/utils/feasibility.py)
- **Inputs**: `unit_row` (`dict` or namedtuple), `trip_ids` (`list` of `str`), `trips_df` (`pd.DataFrame` or `dict`), `dist_lookup` (`dict`).
- **Outputs**: `bool` — `True` if depot departure and all successive arc transitions along the chain are feasible.

### `greedy_construction(instance: dict) -> dict`
- **Location**: [`src/utils/greedy_init.py`](file:///Users/sayantanmandal/Downloads/Quantum-main/src/utils/greedy_init.py)
- **Inputs**: `instance` (`dict`) containing `fleet`, `trips`, `dist_lookup`, and `maint_lookup`.
- **Outputs**: `chains` (`dict` `{unit_id: [trip_id, ...]}`) — Feasible circulation chains constructed via greedy selection and arc feasibility verification.

### `chains_to_schedule_df(chains: dict, trips_df: pd.DataFrame) -> pd.DataFrame`
- **Location**: [`src/utils/greedy_init.py`](file:///Users/sayantanmandal/Downloads/Quantum-main/src/utils/greedy_init.py)
- **Inputs**: `chains` (`dict`), `trips_df` (`pd.DataFrame`).
- **Outputs**: Structured schedule `pd.DataFrame` with columns: `unit_id, trip_id, origin, destination, departure_time, arrival_time`.

### `plot_cost_history(history: list[dict], algo_name: str = "Algorithm", out_path: str = None)`
- **Location**: [`src/utils/viz.py`](file:///Users/sayantanmandal/Downloads/Quantum-main/src/utils/viz.py)
- **Inputs**: `history` (`list` of `dict` with cost trajectories), `algo_name` (`str`), `out_path` (`str`).
- **Outputs**: Exports high-resolution convergence curve PNG to disk.

### `plot_comparison_bar(results: dict, metric: str = "total", out_path: str = None)`
- **Location**: [`src/utils/viz.py`](file:///Users/sayantanmandal/Downloads/Quantum-main/src/utils/viz.py)
- **Inputs**: `results` (`dict` mapping `algo_name -> cost_dict`), `metric` (`str`), `out_path` (`str`).
- **Outputs**: Exports comparative bar chart PNG to disk.

### `plot_failure_heatmap(failure_data: dict, out_path: str)`
- **Location**: [`src/utils/viz.py`](file:///Users/sayantanmandal/Downloads/Quantum-main/src/utils/viz.py)
- **Inputs**: `failure_data` (`dict` mapping `trip_id -> {constraint_type: count}`), `out_path` (`str`).
- **Outputs**: Exports 2D failure heatmap PNG to disk.

---

## 6. How to Run

### Step 1: Install Dependencies
Create and activate a virtual environment (Python 3.10+ recommended) and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Regenerate Instance Data (Optional)
The repository includes a ready-to-run problem instance in `data/`. To regenerate synthetic data:
```bash
python3 generate_data.py
```

### Step 3: Run Master Benchmark (All 4 Algorithms)
To sequentially execute Greedy, Simulated Annealing, Genetic Algorithm, and NSGA-II, generate all schedules, and produce comparative reports:
```bash
python3 src/run_all.py
```

### Step 4: Run Failure Diagnostics and Scaling Simulation
To perform in-depth analysis of classical algorithm limitations and simulate scaling from 20 to 120 trips:
```bash
python3 src/run_failure_analysis.py
```

### Step 5: Run Individual Algorithms Standalone
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

### Step 6: View and Inspect Outputs
- Summary CSV: `outputs/comparison/master_comparison.csv`
- Comparative Charts: `outputs/comparison/master_comparison.png`
- Failure Analysis: `outputs/comparison/failure_summary.json`
- Scaling Curves: `outputs/comparison/scaling_failure.png`

---

## 7. Output Files Reference

| File Path | Description | Format |
| :--- | :--- | :--- |
| `outputs/greedy/final_schedule.csv` | Chronological unit-by-unit trip schedule from greedy heuristic | CSV (`unit_id, trip_id, origin, destination, departure_time, arrival_time`) |
| `outputs/greedy/run_summary.json` | Cost breakdown and execution metadata for greedy baseline | JSON (`total, deadhead_km, units_used, maintenance_violation_km, uncovered_demand, timing_violations`) |
| `outputs/greedy/cost_history.png` | Single-point baseline cost visualization | PNG (High-DPI raster image) |
| `outputs/greedy/failure_analysis.json` | Diagnostic lists of timing violations, uncovered trips, and overloaded trips | JSON (`timing_violations, uncovered_trips, overloaded_trips`) |
| `outputs/sa/final_schedule.csv` | Optimized unit circulation schedule produced by simulated annealing | CSV (`unit_id, trip_id, origin, destination, departure_time, arrival_time`) |
| `outputs/sa/run_summary.json` | Initial vs. final cost breakdown and percentage improvement | JSON (`initial_breakdown, final_breakdown, improvement_pct, hyperparameters`) |
| `outputs/sa/sa_cost_history.csv` | Step-by-step cost and temperature cooling progression | CSV (`step, temp, current_cost, best_cost`) |
| `outputs/sa/cost_history.png` | Simulated annealing convergence curve over cooling steps | PNG (High-DPI raster image) |
| `outputs/ga/final_schedule.csv` | Best circulation schedule discovered by genetic algorithm | CSV (`unit_id, trip_id, origin, destination, departure_time, arrival_time`) |
| `outputs/ga/run_summary.json` | GA hyperparameters, completed generations, and final cost metrics | JSON (`breakdown, generations, hyperparameters, algorithm`) |
| `outputs/ga/ga_cost_history.csv` | Generational log of best and population average costs | CSV (`generation, best_cost, avg_cost`) |
| `outputs/ga/cost_history.png` | Evolutionary cost trajectory plot across generations | PNG (High-DPI raster image) |
| `outputs/ga/failure_analysis.json` | Log of generations with timing violations and infeasible trips | JSON (`generations_where_timing_violations_gt_0, trips_never_feasibly_assigned`) |
| `outputs/nsga2/final_schedule.csv` | Unit circulation schedule for the best compromise Pareto solution | CSV (`unit_id, trip_id, origin, destination, departure_time, arrival_time`) |
| `outputs/nsga2/pareto_front.csv` | Non-dominated objective trade-off records on the final Pareto front | CSV (`deadhead_km, units_used, demand_penalty, rank, generation`) |
| `outputs/nsga2/run_summary.json` | Pareto front cardinality and min/max statistics across objectives | JSON (`n_pareto_solutions, objectives_summary, best_compromise_breakdown`) |
| `outputs/nsga2/cost_history.csv` | Generational progress of best $f_1, f_2, f_3$ and Pareto front size | CSV (`generation, best_f1, best_f2, best_f3, n_pareto`) |
| `outputs/nsga2/cost_history.png` | Convergence plot of primary objective $f_1$ (deadhead km) | PNG (High-DPI raster image) |
| `outputs/nsga2/pareto_front.png` | 3D scatter visualization of $f_1$ vs. $f_2$ vs. $f_3$ colored by crowding distance | PNG (High-DPI raster image) |
| `outputs/comparison/master_comparison.csv` | Unified cross-algorithm benchmark comparison table | CSV (`algorithm, total_cost, deadhead_km, units_used, maintenance_violation_km, uncovered_demand, timing_violations, runtime_sec`) |
| `outputs/comparison/master_comparison.png` | 4-panel comparative bar chart (total cost, deadhead km, units used, uncovered demand) | PNG (High-DPI raster image) |
| `outputs/comparison/failure_summary.json` | Numerical failure comparison between classical Greedy and SA | JSON (`greedy: {...}, sa: {...}`) |
| `outputs/comparison/classical_failure_chart.png` | Side-by-side grouped bar chart with red feasibility threshold line | PNG (High-DPI raster image) |
| `outputs/comparison/cost_reduction_chart.png` | SA cooling trajectory annotated with initial, final, % gain, and GA reference line | PNG (High-DPI raster image) |
| `outputs/comparison/scaling_failure.csv` | Problem scaling benchmark records across 20 to 120 trips | CSV (`n_trips, timing_violations, uncovered_demand, runtime_sec`) |
| `outputs/comparison/scaling_failure.png` | Dual-axis line plot showing greedy failure and timing violations vs. trip scale | PNG (High-DPI raster image) |
