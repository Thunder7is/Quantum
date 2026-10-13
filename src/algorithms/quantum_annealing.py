# Quantum Annealing solver using QUBO formulation for rolling stock scheduling
# Reference: Guth Jarkovsky et al. (2026) - Rolling Stock Planning Using QAOA
# Simulates D-Wave annealer using dimod's SimulatedAnnealingSampler
import os
import sys
import json
import time
import itertools
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import dimod
    from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler
    DIMOD_AVAILABLE = True
except ImportError:
    DIMOD_AVAILABLE = False

try:
    from ..utils.data_loader import load_instance, get_lookups, travel
    from ..utils.cost import compute_cost, DEFAULT_WEIGHTS
    from ..utils.greedy_init import chains_to_schedule_df
    from ..utils.viz import plot_cost_history
except (ImportError, ValueError):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.utils.data_loader import load_instance, get_lookups, travel
    from src.utils.cost import compute_cost, DEFAULT_WEIGHTS
    from src.utils.greedy_init import chains_to_schedule_df
    from src.utils.viz import plot_cost_history

# Output directory for quantum annealing results
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "quantum"))

# ─── QUBO Penalty weights ────────────────────────────────────────────────────
# Lambda values must be large enough to make constraint violations costlier
# than any feasible objective improvement (Lagrange multiplier approach)
LAMBDA_COVERAGE   = 3000.0   # penalty per trip not covered by exactly one unit
LAMBDA_CONTINUITY = 2000.0   # penalty for unit teleporting between stations
LAMBDA_MAINTENANCE= 1000.0   # penalty per km over maintenance threshold
LAMBDA_DEMAND     = 500.0    # penalty per uncovered passenger
W_DEADHEAD        = 1.0      # objective weight for empty km (matches cost.py)

# D-Wave SimulatedAnnealingSampler hyperparameters
NUM_READS         = 300     # number of independent annealing runs
NUM_SWEEPS        = 500     # sweeps per anneal (higher = better quality)
BETA_RANGE        = (0.1, 4.0)  # inverse temperature range (cold = exploitative)
SEED              = 42


# ─── STEP 1: Build feasible arc catalogue ────────────────────────────────────

def build_feasible_arcs(trips_list: list, fleet_list: list, dist_lookup: dict) -> dict:
    """
    For each unit u, enumerate all (trip_i -> trip_j) pairs where unit can
    physically reposition from trip_i destination to trip_j origin in time.
    Returns: {unit_id: set of (trip_i_id, trip_j_id)} feasible arcs.
    Used in: continuity constraint construction.
    """
    arcs = {u["unit_id"]: set() for u in fleet_list}

    for unit in fleet_list:
        uid = unit["unit_id"]
        for t_a in trips_list:
            for t_b in trips_list:
                if t_a["trip_id"] == t_b["trip_id"]:
                    continue
                # Unit can only chain trips in time order
                if t_a["departure_min"] >= t_b["departure_min"]:
                    continue
                # Travel time from t_a destination to t_b origin
                tt, _ = travel(dist_lookup, t_a["destination_station"], t_b["origin_station"])
                earliest_b = t_a["arrival_min"] + t_a["min_turnaround_min"] + tt
                if earliest_b <= t_b["departure_min"]:
                    arcs[uid].add((t_a["trip_id"], t_b["trip_id"]))

    return arcs


def build_initial_availability(unit: dict, trips_list: list, dist_lookup: dict) -> list:
    """
    Returns list of trip_ids a unit can serve as its FIRST trip (from depot).
    Used in: coverage constraint — ensures depot-reachability of first assignment.
    """
    feasible_first = []
    for trip in trips_list:
        tt, _ = travel(dist_lookup, unit["current_location"], trip["origin_station"])
        if unit["available_from_min"] + tt <= trip["departure_min"]:
            feasible_first.append(trip["trip_id"])
    return feasible_first


# ─── STEP 2: Build QUBO binary variables ─────────────────────────────────────

def build_variables(trips_list: list, fleet_list: list) -> dict:
    """
    Binary variable x_{u,t} = 1 means unit u is assigned to trip t.
    Variable index: (unit_id, trip_id) -> integer index in QUBO matrix.
    Returns: (var_index dict, reverse_map list).
    """
    var_index = {}
    reverse_map = []
    idx = 0
    for unit in fleet_list:
        for trip in trips_list:
            key = (unit["unit_id"], trip["trip_id"])
            var_index[key] = idx
            reverse_map.append(key)
            idx += 1
    return var_index, reverse_map


# ─── STEP 3: Objective function terms ────────────────────────────────────────

def add_objective_terms(bqm: "BinaryQuadraticModel", var_index: dict,
                        trips_list: list, fleet_list: list, dist_lookup: dict):
    """
    Objective: minimize total deadhead (empty) kilometers.
    For each variable x_{u,t}, the deadhead cost is the distance unit u
    must travel empty to reach trip t's origin from its current location.

    f_obj = sum_{u,t} [ deadhead_km(u,t) * x_{u,t} ] * W_DEADHEAD

    Connected to: cost.py compute_cost() deadhead_km metric.
    """
    unit_loc = {u["unit_id"]: u["current_location"] for u in fleet_list}

    for unit in fleet_list:
        uid = unit["unit_id"]
        for trip in trips_list:
            tid = trip["trip_id"]
            _, dh_km = travel(dist_lookup, unit_loc[uid], trip["origin_station"])
            cost_val = W_DEADHEAD * dh_km
            key = (uid, tid)
            if key in var_index:
                bqm.add_variable(var_index[key], cost_val)


# ─── STEP 4: Constraint 1 — Trip Coverage ────────────────────────────────────

def add_coverage_constraint(bqm: "BinaryQuadraticModel", var_index: dict,
                            trips_list: list, fleet_list: list,
                            demand_lookup: dict, capacity_lookup: dict):
    """
    Every trip t must be assigned to at least one unit with enough capacity.

    Mathematical formulation:
        sum_{u} x_{u,t} >= 1   for all t in T         (at least one unit)
        sum_{u} x_{u,t} * cap_u >= demand_t            (capacity >= demand)

    Penalty Hamiltonian (QUBO encoding):
        H_cov = LAMBDA_COVERAGE * sum_t ( 1 - sum_u x_{u,t} )^2

    Expanding the square adds linear terms on diagonal and quadratic cross-terms.
    Connected to: compute_cost() uncovered_demand metric.
    """
    for trip in trips_list:
        tid = trip["trip_id"]
        # Indices of all variables for this trip
        trip_vars = [var_index[(u["unit_id"], tid)]
                     for u in fleet_list if (u["unit_id"], tid) in var_index]

        # Linear penalty: -2 * LAMBDA per variable (from expanding (1 - sum x)^2)
        for v in trip_vars:
            bqm.add_variable(v, -2.0 * LAMBDA_COVERAGE)

        # Quadratic penalty: +2 * LAMBDA for every pair (from cross terms)
        for v1, v2 in itertools.combinations(trip_vars, 2):
            bqm.add_interaction(v1, v2, 2.0 * LAMBDA_COVERAGE)

        # Constant term +LAMBDA (from the 1^2 in the expansion) — added to offset
        bqm.offset += LAMBDA_COVERAGE


# ─── STEP 5: Constraint 2 — Continuity ───────────────────────────────────────

def add_continuity_constraint(bqm: "BinaryQuadraticModel", var_index: dict,
                              trips_list: list, fleet_list: list,
                              feasible_arcs: dict):
    """
    If unit u is assigned to trip_i and trip_j consecutively,
    it must be physically possible to travel from trip_i's destination
    to trip_j's origin in time.

    Mathematical formulation:
        x_{u,t_i} * x_{u,t_j} = 0   if (t_i, t_j) not in feasible_arcs[u]

    Penalty Hamiltonian:
        H_cont = LAMBDA_CONTINUITY * sum_{u} sum_{(i,j) infeasible} x_{u,t_i} * x_{u,t_j}

    This penalises any chromosome pairing a unit to two trips it cannot chain.
    Connected to: feasibility.py is_feasible_arc().
    """
    trip_id_set = {t["trip_id"] for t in trips_list}

    for unit in fleet_list:
        uid = unit["unit_id"]
        feasible = feasible_arcs.get(uid, set())

        # For every pair of trips this unit could serve, penalise infeasible chains
        for t_a in trips_list:
            for t_b in trips_list:
                if t_a["trip_id"] == t_b["trip_id"]:
                    continue
                if t_a["departure_min"] >= t_b["departure_min"]:
                    continue
                pair = (t_a["trip_id"], t_b["trip_id"])
                if pair not in feasible:
                    # Add penalty for selecting this infeasible arc
                    k_a = (uid, t_a["trip_id"])
                    k_b = (uid, t_b["trip_id"])
                    if k_a in var_index and k_b in var_index:
                        bqm.add_interaction(
                            var_index[k_a], var_index[k_b],
                            LAMBDA_CONTINUITY
                        )


# ─── STEP 6: Constraint 3 — Maintenance ──────────────────────────────────────

def add_maintenance_constraint(bqm: "BinaryQuadraticModel", var_index: dict,
                               trips_list: list, fleet_list: list,
                               dist_lookup: dict, maint_lookup: dict):
    """
    A unit's total km (km_since_maintenance + trip_km + deadhead_km) must
    not exceed the maintenance threshold.

    Penalty Hamiltonian (linear approximation):
        H_maint = LAMBDA_MAINTENANCE * sum_{u,t} [
            max(0, km_since_maint_u + dh_km(u,t) + trip_km_t - max_km_u)
        ] * x_{u,t}

    We linearise: any trip that would push a unit over its limit gets
    a large linear penalty on its QUBO diagonal.
    Connected to: compute_cost() maintenance_violation_km metric.
    """
    for unit in fleet_list:
        uid = unit["unit_id"]
        max_km = maint_lookup.get(unit["unit_type"], 12000)
        current_km = unit["km_since_maintenance"]

        for trip in trips_list:
            tid = trip["trip_id"]
            _, dh_km = travel(dist_lookup, unit["current_location"], trip["origin_station"])
            total_km = current_km + dh_km + trip["distance_km"]
            if total_km > max_km:
                violation_km = total_km - max_km
                penalty = LAMBDA_MAINTENANCE * violation_km
                key = (uid, tid)
                if key in var_index:
                    bqm.add_variable(var_index[key], penalty)


# ─── STEP 7: Build full QUBO BQM ─────────────────────────────────────────────

def build_qubo(trips_list: list, fleet_list: list, dist_lookup: dict,
               maint_lookup: dict) -> tuple:
    """
    Assembles all QUBO terms into a single BinaryQuadraticModel.
    Returns: (bqm, var_index, reverse_map, feasible_arcs)

    Full Hamiltonian:
        H = H_obj + H_coverage + H_continuity + H_maintenance

    Where H_obj minimises deadhead km and the constraint Hamiltonians
    enforce feasibility via Lagrange penalty parameters.
    """
    if not DIMOD_AVAILABLE:
        raise ImportError("dimod is required. Run: pip install dimod")

    # Step 1: enumerate feasible unit-trip arcs
    feasible_arcs = build_feasible_arcs(trips_list, fleet_list, dist_lookup)

    # Step 2: create binary variables x_{u,t}
    var_index, reverse_map = build_variables(trips_list, fleet_list)

    # Step 3: initialise empty BQM (BINARY type for {0,1} variables)
    bqm = BinaryQuadraticModel(vartype=dimod.BINARY)

    # Build demand and capacity lookups
    demand_lookup = {t["trip_id"]: t.get("expected_passenger_demand", 0) for t in trips_list}
    capacity_lookup = {u["unit_id"]: u["capacity"] for u in fleet_list}

    # Step 4: add objective (deadhead km minimisation)
    add_objective_terms(bqm, var_index, trips_list, fleet_list, dist_lookup)

    # Step 5: add coverage constraint
    add_coverage_constraint(bqm, var_index, trips_list, fleet_list,
                            demand_lookup, capacity_lookup)

    # Step 6: add continuity constraint
    add_continuity_constraint(bqm, var_index, trips_list, fleet_list, feasible_arcs)

    # Step 7: add maintenance constraint
    add_maintenance_constraint(bqm, var_index, trips_list, fleet_list,
                               dist_lookup, maint_lookup)

    print(f"QUBO built: {len(bqm.variables)} variables, "
          f"{len(bqm.quadratic)} quadratic interactions")

    return bqm, var_index, reverse_map, feasible_arcs


# ─── STEP 8: Decode sample → chains ──────────────────────────────────────────

def decode_sample(sample: dict, reverse_map: list, fleet_list: list,
                  trips_list: list) -> dict:
    """
    Converts a QUBO binary sample {var_idx: 0/1} back into chains dict
    {unit_id: [trip_id, ...]} compatible with compute_cost().
    """
    chains = {u["unit_id"]: [] for u in fleet_list}
    trip_id_set = {t["trip_id"] for t in trips_list}

    for var_idx, val in sample.items():
        if val == 1:
            uid, tid = reverse_map[var_idx]
            if tid in trip_id_set:
                chains[uid].append(tid)

    return chains


# ─── STEP 9: D-Wave Simulated Annealing Solver ───────────────────────────────

def solve_quantum_annealing(fleet, trips, dist_lookup, maint_lookup,
                            num_reads=NUM_READS, num_sweeps=NUM_SWEEPS,
                            seed=SEED) -> tuple:
    """
    Main solver: builds QUBO and runs D-Wave SimulatedAnnealingSampler.

    The SimulatedAnnealingSampler from dimod is the classical simulation of
    what D-Wave's QPU does quantum mechanically. It uses the same QUBO
    formulation that would be submitted to actual D-Wave hardware.

    To run on real D-Wave QPU, replace sampler with:
        from dwave.system import DWaveSampler, EmbeddingComposite
        sampler = EmbeddingComposite(DWaveSampler())

    Returns: (best_chains, best_cost, best_breakdown, sample_history)
    """
    if not DIMOD_AVAILABLE:
        raise ImportError("dimod is required. Run: pip install dimod")

    # Prepare list representations of fleet and trips
    trips_sorted = trips.sort_values("departure_min").reset_index(drop=True)
    trips_list = trips_sorted.to_dict("records")
    fleet_list = fleet.to_dict("records")
    fleet_lookup, trips_lookup = get_lookups(fleet, trips_sorted)

    print("Building QUBO formulation...")
    t_build_start = time.time()
    bqm, var_index, reverse_map, feasible_arcs = build_qubo(
        trips_list, fleet_list, dist_lookup, maint_lookup
    )
    t_build_end = time.time()
    print(f"QUBO build time: {t_build_end - t_build_start:.2f}s")

    # Run D-Wave SimulatedAnnealingSampler (classical simulation of QPU)
    print(f"Running SimulatedAnnealingSampler: {num_reads} reads, {num_sweeps} sweeps...")
    sampler = SimulatedAnnealingSampler()

    t_sample_start = time.time()
    response = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_range=BETA_RANGE,
    )
    t_sample_end = time.time()
    sampling_time = t_sample_end - t_sample_start
    print(f"Sampling time: {sampling_time:.2f}s")

    # Collect all samples and evaluate against the original cost function
    sample_history = []
    best_chains = None
    best_cost = float("inf")
    best_breakdown = None

    for i, (sample, energy, num_occur) in enumerate(response.data(["sample", "energy", "num_occurrences"])):
        chains = decode_sample(sample, reverse_map, fleet_list, trips_list)
        breakdown = compute_cost(chains, trips_lookup, fleet_lookup, dist_lookup, maint_lookup)
        real_cost = breakdown["total"]

        sample_history.append({
            "step": i,
            "total_cost": real_cost,
            "best_cost": min(real_cost, best_cost),
            "qubo_energy": energy,
            "num_occurrences": num_occur,
        })

        if real_cost < best_cost:
            best_cost = real_cost
            best_chains = chains
            best_breakdown = breakdown

    # Update best_cost in history
    running_best = float("inf")
    for rec in sample_history:
        if rec["total_cost"] < running_best:
            running_best = rec["total_cost"]
        rec["best_cost"] = running_best

    return best_chains, best_cost, best_breakdown, sample_history, sampling_time


# ─── STEP 10: QUBO Statistics and Analysis ───────────────────────────────────

def analyse_qubo(bqm: "BinaryQuadraticModel", var_index: dict,
                 trips_list: list, fleet_list: list) -> dict:
    """
    Computes QUBO matrix statistics for the research paper.
    Reports density, variable count, interaction count, and energy range.
    """
    n_vars = len(bqm.variables)
    n_interactions = len(bqm.quadratic)
    max_possible_interactions = n_vars * (n_vars - 1) / 2
    density = n_interactions / max_possible_interactions if max_possible_interactions > 0 else 0

    linear_vals = list(bqm.linear.values())
    quad_vals = list(bqm.quadratic.values())

    return {
        "n_variables": n_vars,
        "n_qubits_needed": n_vars,
        "n_trips": len(trips_list),
        "n_units": len(fleet_list),
        "n_quadratic_interactions": n_interactions,
        "qubo_density": round(density, 4),
        "linear_min": round(min(linear_vals), 2) if linear_vals else 0,
        "linear_max": round(max(linear_vals), 2) if linear_vals else 0,
        "quadratic_min": round(min(quad_vals), 2) if quad_vals else 0,
        "quadratic_max": round(max(quad_vals), 2) if quad_vals else 0,
        "offset": round(bqm.offset, 2),
        "penalty_weights": {
            "lambda_coverage": LAMBDA_COVERAGE,
            "lambda_continuity": LAMBDA_CONTINUITY,
            "lambda_maintenance": LAMBDA_MAINTENANCE,
            "lambda_demand": LAMBDA_DEMAND,
            "w_deadhead": W_DEADHEAD,
        }
    }


# ─── STEP 11: Visualisations ─────────────────────────────────────────────────

def plot_qubo_energy_landscape(sample_history: list, out_path: str):
    """
    Plots QUBO energy vs sample number and real cost convergence.
    Shows how the annealer traverses the energy landscape.
    """
    steps = [r["step"] for r in sample_history]
    qubo_energy = [r["qubo_energy"] for r in sample_history]
    real_cost = [r["total_cost"] for r in sample_history]
    best_cost = [r["best_cost"] for r in sample_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # QUBO energy landscape
    ax1.plot(steps, qubo_energy, alpha=0.4, color="#9b59b6", linewidth=0.8, label="QUBO Energy")
    ax1.set_xlabel("Sample Number")
    ax1.set_ylabel("QUBO Energy (Hamiltonian value)")
    ax1.set_title("Quantum Annealing — QUBO Energy Landscape Traversal")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Real cost convergence
    ax2.plot(steps, real_cost, alpha=0.3, color="#3498db", linewidth=0.8, label="Sample Cost")
    ax2.plot(steps, best_cost, color="#e74c3c", linewidth=2.0, label="Best Cost Found")
    ax2.set_xlabel("Sample Number")
    ax2.set_ylabel("Total Operational Cost")
    ax2.set_title("Quantum Annealing — Real Cost Convergence Across Samples")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_qubo_structure(bqm: "BinaryQuadraticModel", var_index: dict,
                        trips_list: list, fleet_list: list, out_path: str):
    """
    Visualises the QUBO matrix sparsity structure.
    Shows which variables interact (non-zero off-diagonal entries).
    Used in research paper to demonstrate QUBO encoding complexity.
    """
    n = len(bqm.variables)
    # Build dense matrix for small instances only
    if n > 500:
        print(f"QUBO matrix ({n}x{n}) too large for full visualisation. Skipping.")
        return

    Q = np.zeros((n, n))
    for (i, j), val in bqm.quadratic.items():
        Q[i][j] = val
        Q[j][i] = val
    for i, val in bqm.linear.items():
        Q[i][i] = val

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(Q != 0, cmap="Blues", aspect="auto", interpolation="none")
    ax.set_title(f"QUBO Matrix Sparsity Structure\n"
                 f"{n} variables | {len(bqm.quadratic)} interactions | "
                 f"{len(trips_list)} trips x {len(fleet_list)} units",
                 fontsize=11)
    ax.set_xlabel("Variable Index (unit × trip)")
    ax.set_ylabel("Variable Index (unit × trip)")
    plt.colorbar(im, ax=ax, label="Non-zero entry")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_energy_histogram(sample_history: list, out_path: str):
    """
    Distribution of QUBO energies across all samples.
    Lower energy = better solution. Shows quality of annealing.
    """
    energies = [r["qubo_energy"] for r in sample_history]
    costs = [r["total_cost"] for r in sample_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(energies, bins=40, color="#9b59b6", alpha=0.75, edgecolor="black")
    ax1.set_xlabel("QUBO Energy")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of QUBO Energies Across Samples")
    ax1.axvline(min(energies), color="red", linestyle="--", label=f"Min: {min(energies):.0f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(costs, bins=40, color="#3498db", alpha=0.75, edgecolor="black")
    ax2.set_xlabel("Total Operational Cost")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Real Costs Across Samples")
    ax2.axvline(min(costs), color="red", linestyle="--", label=f"Min: {min(costs):.0f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Quantum Annealing — Sample Quality Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── STEP 12: Main Execution ──────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not DIMOD_AVAILABLE:
        print("ERROR: dimod not installed. Run: pip install dimod")
        return

    inst = load_instance()
    fleet = inst["fleet"]
    trips = inst["trips"]
    dist_lookup = inst["dist_lookup"]
    maint_lookup = inst["maint_lookup"]
    fleet_lookup, trips_lookup = get_lookups(fleet, trips)

    print("=================================================================")
    print("  Quantum Annealing Solver (QUBO + D-Wave Simulation)")
    print(f"  Fleet: {len(fleet)} units | Trips: {len(trips)} | Stations: 5")
    print(f"  Sampler: dimod SimulatedAnnealingSampler (D-Wave equivalent)")
    print(f"  Reads: {NUM_READS} | Sweeps: {NUM_SWEEPS}")
    print("=================================================================\n")

    # Build QUBO for analysis stats before solving
    trips_list = trips.sort_values("departure_min").reset_index(drop=True).to_dict("records")
    fleet_list = fleet.to_dict("records")

    print("Analysing QUBO structure...")
    feasible_arcs = build_feasible_arcs(trips_list, fleet_list, dist_lookup)
    var_index, reverse_map = build_variables(trips_list, fleet_list)
    bqm_temp = build_qubo(trips_list, fleet_list, dist_lookup, maint_lookup)[0]
    qubo_stats = analyse_qubo(bqm_temp, var_index, trips_list, fleet_list)

    print("\nQUBO Statistics:")
    print(json.dumps(qubo_stats, indent=2))
    del bqm_temp  # free memory before solving

    # Run quantum annealing solver
    best_chains, best_cost, best_breakdown, sample_history, sampling_time = solve_quantum_annealing(
        fleet, trips, dist_lookup, maint_lookup,
        num_reads=NUM_READS, num_sweeps=NUM_SWEEPS, seed=SEED
    )

    print("\nQuantum Annealing best solution breakdown:")
    print(json.dumps(best_breakdown, indent=2))

    # ── Save all outputs ──────────────────────────────────────────────────────

    # Final schedule CSV
    schedule_df = chains_to_schedule_df(best_chains, trips)
    schedule_path = os.path.join(OUT_DIR, "final_schedule.csv")
    schedule_df.to_csv(schedule_path, index=False)

    # Sample history CSV
    history_df = pd.DataFrame(sample_history)
    history_path = os.path.join(OUT_DIR, "qa_sample_history.csv")
    history_df.to_csv(history_path, index=False)

    # Run summary JSON
    run_summary = {
        "breakdown": best_breakdown,
        "qubo_statistics": qubo_stats,
        "hyperparameters": {
            "num_reads": NUM_READS,
            "num_sweeps": NUM_SWEEPS,
            "beta_range": list(BETA_RANGE),
            "lambda_coverage": LAMBDA_COVERAGE,
            "lambda_continuity": LAMBDA_CONTINUITY,
            "lambda_maintenance": LAMBDA_MAINTENANCE,
            "w_deadhead": W_DEADHEAD,
        },
        "sampling_time_sec": round(sampling_time, 3),
        "sampler": "dimod.SimulatedAnnealingSampler (D-Wave classical simulation)",
        "algorithm": "quantum_annealing_qubo",
        "paper_reference": "Guth Jarkovsky et al. 2026 - adapted for Indian Railway Network",
    }
    summary_path = os.path.join(OUT_DIR, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)

    # QUBO stats JSON (standalone for paper reference)
    qubo_path = os.path.join(OUT_DIR, "qubo_statistics.json")
    with open(qubo_path, "w") as f:
        json.dump(qubo_stats, f, indent=2)

    # Plots
    energy_plot = os.path.join(OUT_DIR, "qa_energy_landscape.png")
    plot_qubo_energy_landscape(sample_history, energy_plot)

    histogram_plot = os.path.join(OUT_DIR, "qa_energy_histogram.png")
    plot_energy_histogram(sample_history, histogram_plot)

    # QUBO structure plot (only if small enough)
    _, var_index_full, _, _ = build_qubo(trips_list, fleet_list, dist_lookup, maint_lookup)
    structure_plot = os.path.join(OUT_DIR, "qubo_matrix_structure.png")
    bqm_for_plot, var_index_plot, _, _ = build_qubo(trips_list, fleet_list, dist_lookup, maint_lookup)
    plot_qubo_structure(bqm_for_plot, var_index_plot, trips_list, fleet_list, structure_plot)

    # Cost convergence chart (reuse plot_cost_history from viz.py)
    cost_plot = os.path.join(OUT_DIR, "cost_history.png")
    plot_cost_history(sample_history, "Quantum Annealing (QUBO)", cost_plot)

    print(f"\nSaved schedule:        {schedule_path}")
    print(f"Saved sample history:  {history_path}")
    print(f"Saved run summary:     {summary_path}")
    print(f"Saved QUBO stats:      {qubo_path}")
    print(f"Saved energy plot:     {energy_plot}")
    print(f"Saved histogram:       {histogram_plot}")
    print(f"Saved structure plot:  {structure_plot}")
    print(f"Saved cost history:    {cost_plot}")


if __name__ == "__main__":
    main()