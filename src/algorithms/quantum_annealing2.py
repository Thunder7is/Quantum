# Quantum Annealing solver using QUBO formulation for rolling stock scheduling
# Reference: Guth Jarkovsky et al. (2026) - Rolling Stock Planning Using QAOA
# Primary solver: D-Wave TabuSampler (quantum-inspired, best classical QUBO solver)
# Fallback solver: neal.SimulatedAnnealingSampler (C++ optimised)
# Final fallback: dimod.SimulatedAnnealingSampler (pure Python)
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

# ── Sampler imports with graceful fallback chain ──────────────────────────────
try:
    from tabu import TabuSampler
    TABU_AVAILABLE = True
except ImportError:
    TABU_AVAILABLE = False

try:
    import neal
    NEAL_AVAILABLE = True
except ImportError:
    NEAL_AVAILABLE = False

try:
    import dimod
    from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler as DimodSASampler
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
OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "quantum2")
)

# ─── QUBO Penalty weights (Lagrange multipliers) ─────────────────────────────
LAMBDA_COVERAGE    = 3000.0   # penalty per trip not covered by exactly one unit
LAMBDA_CONTINUITY  = 2000.0   # penalty for infeasible unit-to-unit arc
LAMBDA_MAINTENANCE = 1000.0   # penalty per km over maintenance threshold
LAMBDA_DEMAND      = 500.0    # penalty per uncovered passenger
W_DEADHEAD         = 1.0      # objective weight for empty km (matches cost.py)

# ─── TabuSampler hyperparameters ─────────────────────────────────────────────
# TabuSampler uses tabu search which mimics quantum tunneling classically.
# tenure: number of moves each variable is forbidden after being flipped.
# timeout: total wall-clock seconds the sampler runs (controls runtime directly).
# num_reads: independent tabu restarts (more = better quality, more time).
TABU_NUM_READS = 5500       # independent restarts
TABU_TENURE    = 30         # tabu tenure (10-30 is optimal for QUBO)
TABU_TIMEOUT   = 1500       # seconds total (5 minutes — fast on M1)

# ─── Neal fallback hyperparameters ───────────────────────────────────────────
NEAL_NUM_READS  = 500
NEAL_NUM_SWEEPS = 3000      # more sweeps than dimod default, better quality

# ─── Dimod fallback hyperparameters ──────────────────────────────────────────
DIMOD_NUM_READS  = 300
DIMOD_NUM_SWEEPS = 500
BETA_RANGE       = (0.1, 4.0)

SEED = 42


# ─── STEP 1: Build feasible arc catalogue ────────────────────────────────────

def build_feasible_arcs(trips_list: list, fleet_list: list,
                        dist_lookup: dict) -> dict:
    """
    For each unit u, find all (trip_i, trip_j) pairs where u can
    reposition from trip_i destination to trip_j origin in time.
    Returns: {unit_id: set of (trip_i_id, trip_j_id)}.
    Used in: continuity constraint — is_feasible_arc() equivalent in QUBO.
    """
    arcs = {u["unit_id"]: set() for u in fleet_list}
    for unit in fleet_list:
        uid = unit["unit_id"]
        for t_a in trips_list:
            for t_b in trips_list:
                if t_a["trip_id"] == t_b["trip_id"]:
                    continue
                if t_a["departure_min"] >= t_b["departure_min"]:
                    continue
                tt, _ = travel(dist_lookup, t_a["destination_station"],
                               t_b["origin_station"])
                earliest_b = t_a["arrival_min"] + t_a["min_turnaround_min"] + tt
                if earliest_b <= t_b["departure_min"]:
                    arcs[uid].add((t_a["trip_id"], t_b["trip_id"]))
    return arcs


# ─── STEP 2: Binary variable index ───────────────────────────────────────────

def build_variables(trips_list: list, fleet_list: list) -> tuple:
    """
    x_{u,t} = 1 means unit u is assigned to trip t.
    Returns: (var_index dict, reverse_map list).
    var_index: (unit_id, trip_id) -> integer QUBO variable index.
    reverse_map: integer index -> (unit_id, trip_id).
    """
    var_index, reverse_map = {}, []
    idx = 0
    for unit in fleet_list:
        for trip in trips_list:
            key = (unit["unit_id"], trip["trip_id"])
            var_index[key] = idx
            reverse_map.append(key)
            idx += 1
    return var_index, reverse_map


# ─── STEP 3: Objective — minimise deadhead km ────────────────────────────────

def add_objective_terms(bqm, var_index: dict, trips_list: list,
                        fleet_list: list, dist_lookup: dict):
    """
    f_obj = sum_{u,t} deadhead_km(u -> origin_t) * x_{u,t} * W_DEADHEAD

    Adds linear diagonal terms to BQM.
    Connected to: cost.py compute_cost() deadhead_km metric.
    """
    unit_loc = {u["unit_id"]: u["current_location"] for u in fleet_list}
    for unit in fleet_list:
        uid = unit["unit_id"]
        for trip in trips_list:
            tid = trip["trip_id"]
            _, dh_km = travel(dist_lookup, unit_loc[uid], trip["origin_station"])
            key = (uid, tid)
            if key in var_index:
                bqm.add_variable(var_index[key], W_DEADHEAD * dh_km)


# ─── STEP 4: Constraint 1 — Trip Coverage ────────────────────────────────────

def add_coverage_constraint(bqm, var_index: dict, trips_list: list,
                            fleet_list: list):
    """
    Every trip must be assigned to exactly one unit.

    Penalty Hamiltonian:
        H_cov = LAMBDA_COVERAGE * sum_t (1 - sum_u x_{u,t})^2

    Expanding:
        = LAMBDA * [1 - 2*sum_u x_{u,t} + (sum_u x_{u,t})^2]
        = LAMBDA * [1 - 2*sum_u x_{u,t} + sum_u x_{u,t}^2
                      + 2*sum_{u<v} x_{u,t}*x_{v,t}]

    Since x^2 = x for binary variables:
        linear  = -2 * LAMBDA per variable
        quadratic = +2 * LAMBDA per pair of units on same trip
        offset += LAMBDA per trip

    Connected to: compute_cost() uncovered_demand metric.
    """
    for trip in trips_list:
        tid = trip["trip_id"]
        trip_vars = [var_index[(u["unit_id"], tid)]
                     for u in fleet_list if (u["unit_id"], tid) in var_index]
        for v in trip_vars:
            bqm.add_variable(v, -2.0 * LAMBDA_COVERAGE)
        for v1, v2 in itertools.combinations(trip_vars, 2):
            bqm.add_interaction(v1, v2, 2.0 * LAMBDA_COVERAGE)
        bqm.offset += LAMBDA_COVERAGE


# ─── STEP 5: Constraint 2 — Continuity ───────────────────────────────────────

def add_continuity_constraint(bqm, var_index: dict, trips_list: list,
                              fleet_list: list, feasible_arcs: dict):
    """
    Penalise any unit assigned to two trips it cannot physically chain.

    Penalty Hamiltonian:
        H_cont = LAMBDA_CONTINUITY * sum_u sum_{(i,j) infeasible} x_{u,i} * x_{u,j}

    Adds quadratic interaction for each infeasible (unit, trip_i, trip_j) triple.
    Connected to: feasibility.py is_feasible_arc().
    """
    for unit in fleet_list:
        uid = unit["unit_id"]
        feasible = feasible_arcs.get(uid, set())
        for t_a in trips_list:
            for t_b in trips_list:
                if t_a["trip_id"] == t_b["trip_id"]:
                    continue
                if t_a["departure_min"] >= t_b["departure_min"]:
                    continue
                if (t_a["trip_id"], t_b["trip_id"]) not in feasible:
                    k_a = (uid, t_a["trip_id"])
                    k_b = (uid, t_b["trip_id"])
                    if k_a in var_index and k_b in var_index:
                        bqm.add_interaction(
                            var_index[k_a], var_index[k_b],
                            LAMBDA_CONTINUITY
                        )


# ─── STEP 6: Constraint 3 — Maintenance ──────────────────────────────────────

def add_maintenance_constraint(bqm, var_index: dict, trips_list: list,
                               fleet_list: list, dist_lookup: dict,
                               maint_lookup: dict):
    """
    Penalise assigning a trip to a unit that would exceed its km limit.

    Linear penalty (per variable):
        H_maint = LAMBDA_MAINTENANCE
                  * max(0, km_since_maint_u + dh_km + trip_km - max_km_u)
                  * x_{u,t}

    Connected to: compute_cost() maintenance_violation_km metric.
    """
    for unit in fleet_list:
        uid = unit["unit_id"]
        max_km = maint_lookup.get(unit["unit_type"], 12000)
        current_km = unit["km_since_maintenance"]
        for trip in trips_list:
            tid = trip["trip_id"]
            _, dh_km = travel(dist_lookup, unit["current_location"],
                              trip["origin_station"])
            total_km = current_km + dh_km + trip["distance_km"]
            if total_km > max_km:
                violation_km = total_km - max_km
                key = (uid, tid)
                if key in var_index:
                    bqm.add_variable(var_index[key],
                                     LAMBDA_MAINTENANCE * violation_km)


# ─── STEP 7: Assemble full QUBO BQM ──────────────────────────────────────────

def build_qubo(trips_list: list, fleet_list: list, dist_lookup: dict,
               maint_lookup: dict) -> tuple:
    """
    Full Hamiltonian:
        H = H_obj + H_coverage + H_continuity + H_maintenance

    Returns: (bqm, var_index, reverse_map, feasible_arcs).
    All constraint terms use Lagrange penalty parameters so that any
    feasibility violation costs more than the maximum possible
    objective improvement.
    """
    if not DIMOD_AVAILABLE:
        raise ImportError("dimod is required. Run: pip install dimod")

    feasible_arcs = build_feasible_arcs(trips_list, fleet_list, dist_lookup)
    var_index, reverse_map = build_variables(trips_list, fleet_list)
    bqm = BinaryQuadraticModel(vartype=dimod.BINARY)

    add_objective_terms(bqm, var_index, trips_list, fleet_list, dist_lookup)
    add_coverage_constraint(bqm, var_index, trips_list, fleet_list)
    add_continuity_constraint(bqm, var_index, trips_list, fleet_list,
                              feasible_arcs)
    add_maintenance_constraint(bqm, var_index, trips_list, fleet_list,
                               dist_lookup, maint_lookup)

    print(f"QUBO built: {len(bqm.variables)} variables, "
          f"{len(bqm.quadratic)} quadratic interactions")

    return bqm, var_index, reverse_map, feasible_arcs


# ─── STEP 8: Decode sample → chains ──────────────────────────────────────────

def decode_sample(sample: dict, reverse_map: list, fleet_list: list,
                  trips_list: list) -> dict:
    """
    Convert QUBO binary sample {var_idx: 0/1} to chains dict
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


# ─── STEP 9: Pick best available sampler ─────────────────────────────────────

def _get_sampler_info() -> tuple:
    """
    Returns (sampler_name, sampler_description) for the best available solver.
    Priority: TabuSampler > neal.SimulatedAnnealingSampler > dimod.SimulatedAnnealingSampler
    """
    if TABU_AVAILABLE:
        return ("TabuSampler",
                "D-Wave TabuSampler — quantum-inspired tabu search (best classical QUBO solver)")
    if NEAL_AVAILABLE:
        return ("NealSampler",
                "neal.SimulatedAnnealingSampler — C++ optimised simulated annealing")
    return ("DimodSASampler",
            "dimod.SimulatedAnnealingSampler — pure Python fallback")


def _run_sampler(bqm) -> tuple:
    """
    Runs the best available sampler and returns (response, sampler_name, num_reads).

    TabuSampler:
        Uses tabu search — a quantum-inspired method that maintains a list
        of recently visited states (tabu list) to avoid revisiting them.
        This mimics quantum tunneling by forcibly escaping local optima.
        Much better than SA for large sparse QUBO problems like ours.

    Neal:
        C++ implementation of SA — same algorithm as dimod but 5-10x faster.
        Better solution quality per unit time than dimod pure Python.

    Dimod:
        Pure Python SA — slowest but always available as fallback.
    """
    if TABU_AVAILABLE:
        sampler = TabuSampler()
        response = sampler.sample(
            bqm,
            num_reads=TABU_NUM_READS,
            tenure=TABU_TENURE,
            timeout=TABU_TIMEOUT,
        )
        return response, "TabuSampler", TABU_NUM_READS

    if NEAL_AVAILABLE:
        sampler = neal.SimulatedAnnealingSampler()
        response = sampler.sample(
            bqm,
            num_reads=NEAL_NUM_READS,
            num_sweeps=NEAL_NUM_SWEEPS,
            seed=SEED,
        )
        return response, "NealSampler", NEAL_NUM_READS

    # Pure Python dimod fallback
    sampler = DimodSASampler()
    response = sampler.sample(
        bqm,
        num_reads=DIMOD_NUM_READS,
        num_sweeps=DIMOD_NUM_SWEEPS,
        beta_range=BETA_RANGE,
        seed=SEED,
    )
    return response, "DimodSASampler", DIMOD_NUM_READS


# ─── STEP 10: Main Solver ─────────────────────────────────────────────────────

def solve_quantum_annealing(fleet, trips, dist_lookup, maint_lookup,
                            num_reads=None, num_sweeps=None,
                            seed=SEED) -> tuple:
    """
    Builds QUBO and runs the best available QUBO solver.

    TabuSampler is the primary solver — D-Wave's own quantum-inspired
    classical solver. It uses tabu search to escape local optima in a
    way that approximates quantum tunneling behaviour, making it the
    best option on classical hardware before actual QPU access.

    To switch to real D-Wave QPU, replace _run_sampler() with:
        from dwave.system import DWaveSampler, EmbeddingComposite
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample(bqm, num_reads=1000, annealing_time=20)

    Returns: (best_chains, best_cost, best_breakdown, sample_history, sampling_time)
    """
    if not DIMOD_AVAILABLE:
        raise ImportError("dimod is required. Run: pip install dimod")

    trips_sorted = trips.sort_values("departure_min").reset_index(drop=True)
    trips_list   = trips_sorted.to_dict("records")
    fleet_list   = fleet.to_dict("records")
    fleet_lookup, trips_lookup = get_lookups(fleet, trips_sorted)

    # Build QUBO once — reused by solver and analysis
    print("Building QUBO formulation...")
    t0 = time.time()
    bqm, var_index, reverse_map, feasible_arcs = build_qubo(
        trips_list, fleet_list, dist_lookup, maint_lookup
    )
    print(f"QUBO build time: {time.time() - t0:.2f}s")

    # Select and run best available sampler
    sampler_name, sampler_desc = _get_sampler_info()
    print(f"Sampler: {sampler_desc}")

    if TABU_AVAILABLE:
        print(f"TabuSampler config: {TABU_NUM_READS} reads | "
              f"tenure={TABU_TENURE} | timeout={TABU_TIMEOUT}s")
    elif NEAL_AVAILABLE:
        print(f"Neal config: {NEAL_NUM_READS} reads | {NEAL_NUM_SWEEPS} sweeps")
    else:
        print(f"Dimod SA config: {DIMOD_NUM_READS} reads | {DIMOD_NUM_SWEEPS} sweeps")

    t_sample = time.time()
    response, sampler_used, n_reads = _run_sampler(bqm)
    sampling_time = time.time() - t_sample
    print(f"Sampling time: {sampling_time:.2f}s")

    # Evaluate all samples against the real cost function
    sample_history = []
    best_chains    = None
    best_cost      = float("inf")
    best_breakdown = None

    for i, (sample, energy, num_occur) in enumerate(
            response.data(["sample", "energy", "num_occurrences"])):
        chains    = decode_sample(sample, reverse_map, fleet_list, trips_list)
        breakdown = compute_cost(chains, trips_lookup, fleet_lookup,
                                 dist_lookup, maint_lookup)
        real_cost = breakdown["total"]

        sample_history.append({
            "step":            i,
            "total_cost":      real_cost,
            "best_cost":       min(real_cost, best_cost),
            "qubo_energy":     energy,
            "num_occurrences": num_occur,
        })

        if real_cost < best_cost:
            best_cost      = real_cost
            best_chains    = chains
            best_breakdown = breakdown

    # Fix running best in history
    running_best = float("inf")
    for rec in sample_history:
        if rec["total_cost"] < running_best:
            running_best = rec["total_cost"]
        rec["best_cost"] = running_best

    return best_chains, best_cost, best_breakdown, sample_history, sampling_time


# ─── STEP 11: QUBO Statistics ────────────────────────────────────────────────

def analyse_qubo(bqm, var_index: dict, trips_list: list,
                 fleet_list: list) -> dict:
    """
    Computes QUBO matrix statistics for the research paper.
    Reports density, variable count, interaction count, and value ranges.
    """
    n_vars       = len(bqm.variables)
    n_interact   = len(bqm.quadratic)
    max_possible = n_vars * (n_vars - 1) / 2
    density      = n_interact / max_possible if max_possible > 0 else 0

    linear_vals = list(bqm.linear.values())
    quad_vals   = list(bqm.quadratic.values())

    return {
        "n_variables":              n_vars,
        "n_qubits_needed":          n_vars,
        "n_trips":                  len(trips_list),
        "n_units":                  len(fleet_list),
        "n_quadratic_interactions": n_interact,
        "qubo_density":             round(density, 4),
        "linear_min":               round(min(linear_vals), 2) if linear_vals else 0,
        "linear_max":               round(max(linear_vals), 2) if linear_vals else 0,
        "quadratic_min":            round(min(quad_vals), 2) if quad_vals else 0,
        "quadratic_max":            round(max(quad_vals), 2) if quad_vals else 0,
        "offset":                   round(bqm.offset, 2),
        "penalty_weights": {
            "lambda_coverage":    LAMBDA_COVERAGE,
            "lambda_continuity":  LAMBDA_CONTINUITY,
            "lambda_maintenance": LAMBDA_MAINTENANCE,
            "lambda_demand":      LAMBDA_DEMAND,
            "w_deadhead":         W_DEADHEAD,
        },
    }


# ─── STEP 12: Visualisations ──────────────────────────────────────────────────

def plot_qubo_energy_landscape(sample_history: list, out_path: str,
                               sampler_name: str = "Quantum Annealing"):
    """
    Top panel: QUBO Hamiltonian energy across samples.
    Bottom panel: Real operational cost — sample cost + best cost so far.
    """
    steps       = [r["step"]       for r in sample_history]
    qubo_energy = [r["qubo_energy"] for r in sample_history]
    real_cost   = [r["total_cost"] for r in sample_history]
    best_cost   = [r["best_cost"]  for r in sample_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(steps, qubo_energy, alpha=0.5, color="#9b59b6",
             linewidth=0.9, label="QUBO Energy")
    ax1.set_xlabel("Sample Number")
    ax1.set_ylabel("QUBO Energy (Hamiltonian)")
    ax1.set_title(f"{sampler_name} — QUBO Energy Landscape Traversal")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(steps, real_cost, alpha=0.25, color="#3498db",
             linewidth=0.8, label="Sample Cost")
    ax2.plot(steps, best_cost, color="#e74c3c",
             linewidth=2.2, label="Best Cost Found")
    ax2.axhline(best_cost[-1], color="#e74c3c", linestyle=":",
                alpha=0.5, label=f"Final Best: {best_cost[-1]:,.0f}")
    ax2.set_xlabel("Sample Number")
    ax2.set_ylabel("Total Operational Cost")
    ax2.set_title(f"{sampler_name} — Real Cost Convergence")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_energy_histogram(sample_history: list, out_path: str,
                          sampler_name: str = "Quantum Annealing"):
    """
    Left: distribution of QUBO energies across samples.
    Right: distribution of real operational costs.
    Both show the quality spread of the annealer's exploration.
    """
    energies = [r["qubo_energy"] for r in sample_history]
    costs    = [r["total_cost"]  for r in sample_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(energies, bins=40, color="#9b59b6", alpha=0.75, edgecolor="black")
    ax1.axvline(min(energies), color="red", linestyle="--",
                label=f"Min: {min(energies):,.0f}")
    ax1.set_xlabel("QUBO Energy")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Distribution of QUBO Energies — {sampler_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(costs, bins=40, color="#3498db", alpha=0.75, edgecolor="black")
    ax2.axvline(min(costs), color="red", linestyle="--",
                label=f"Min: {min(costs):,.0f}")
    ax2.set_xlabel("Total Operational Cost")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Distribution of Real Costs — {sampler_name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"{sampler_name} — Sample Quality Distribution",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_qubo_structure(bqm, var_index: dict, trips_list: list,
                        fleet_list: list, out_path: str):
    """
    Sparsity pattern of the QUBO matrix.
    Only plotted when n_variables <= 500 (small instances).
    Used in research paper to illustrate QUBO encoding complexity.
    """
    n = len(bqm.variables)
    if n > 500:
        print(f"QUBO matrix ({n}x{n}) too large for visualisation — skipping.")
        return

    Q = np.zeros((n, n))
    for (i, j), val in bqm.quadratic.items():
        Q[i][j] = val
        Q[j][i] = val
    for i, val in bqm.linear.items():
        Q[i][i] = val

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(Q != 0, cmap="Blues", aspect="auto", interpolation="none")
    ax.set_title(
        f"QUBO Matrix Sparsity Structure\n"
        f"{n} variables | {len(bqm.quadratic)} interactions | "
        f"{len(trips_list)} trips × {len(fleet_list)} units",
        fontsize=11
    )
    ax.set_xlabel("Variable Index (unit × trip)")
    ax.set_ylabel("Variable Index (unit × trip)")
    plt.colorbar(im, ax=ax, label="Non-zero entry")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── STEP 13: Main Execution ──────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not DIMOD_AVAILABLE:
        print("ERROR: dimod not installed. Run: pip install dimod")
        return

    sampler_name, sampler_desc = _get_sampler_info()

    inst = load_instance()
    fleet        = inst["fleet"]
    trips        = inst["trips"]
    dist_lookup  = inst["dist_lookup"]
    maint_lookup = inst["maint_lookup"]
    fleet_lookup, trips_lookup = get_lookups(fleet, trips)

    print("=================================================================")
    print("  Quantum Annealing Solver (QUBO + D-Wave Simulation)")
    print(f"  Fleet: {len(fleet)} units | Trips: {len(trips)} | Stations: 5")
    print(f"  Sampler: {sampler_desc}")
    if TABU_AVAILABLE:
        print(f"  Reads: {TABU_NUM_READS} | Tenure: {TABU_TENURE} | Timeout: {TABU_TIMEOUT}s")
    elif NEAL_AVAILABLE:
        print(f"  Reads: {NEAL_NUM_READS} | Sweeps: {NEAL_NUM_SWEEPS}")
    else:
        print(f"  Reads: {DIMOD_NUM_READS} | Sweeps: {DIMOD_NUM_SWEEPS}")
    print("=================================================================\n")

    # Build QUBO once for analysis stats
    trips_sorted = trips.sort_values("departure_min").reset_index(drop=True)
    trips_list   = trips_sorted.to_dict("records")
    fleet_list   = fleet.to_dict("records")

    print("Analysing QUBO structure...")
    bqm_stats, var_index, reverse_map, _ = build_qubo(
        trips_list, fleet_list, dist_lookup, maint_lookup
    )
    qubo_stats = analyse_qubo(bqm_stats, var_index, trips_list, fleet_list)
    del bqm_stats  # free memory before solver runs

    print("\nQUBO Statistics:")
    print(json.dumps(qubo_stats, indent=2))

    # Run solver
    (best_chains, best_cost, best_breakdown,
     sample_history, sampling_time) = solve_quantum_annealing(
        fleet, trips, dist_lookup, maint_lookup, seed=SEED
    )

    print("\nQuantum Annealing best solution breakdown:")
    print(json.dumps(best_breakdown, indent=2))

    # ── Save outputs ──────────────────────────────────────────────────────────

    # Final schedule
    schedule_df   = chains_to_schedule_df(best_chains, trips)
    schedule_path = os.path.join(OUT_DIR, "final_schedule.csv")
    schedule_df.to_csv(schedule_path, index=False)

    # Sample history CSV
    history_df   = pd.DataFrame(sample_history)
    history_path = os.path.join(OUT_DIR, "qa_sample_history.csv")
    history_df.to_csv(history_path, index=False)

    # Run summary JSON
    run_summary = {
        "breakdown": best_breakdown,
        "qubo_statistics": qubo_stats,
        "hyperparameters": {
            "sampler": sampler_name,
            "tabu_num_reads":    TABU_NUM_READS    if TABU_AVAILABLE  else None,
            "tabu_tenure":       TABU_TENURE       if TABU_AVAILABLE  else None,
            "tabu_timeout_sec":  TABU_TIMEOUT      if TABU_AVAILABLE  else None,
            "neal_num_reads":    NEAL_NUM_READS     if NEAL_AVAILABLE  else None,
            "neal_num_sweeps":   NEAL_NUM_SWEEPS    if NEAL_AVAILABLE  else None,
            "dimod_num_reads":   DIMOD_NUM_READS    if not (TABU_AVAILABLE or NEAL_AVAILABLE) else None,
            "dimod_num_sweeps":  DIMOD_NUM_SWEEPS   if not (TABU_AVAILABLE or NEAL_AVAILABLE) else None,
            "lambda_coverage":   LAMBDA_COVERAGE,
            "lambda_continuity": LAMBDA_CONTINUITY,
            "lambda_maintenance":LAMBDA_MAINTENANCE,
            "w_deadhead":        W_DEADHEAD,
        },
        "sampling_time_sec": round(sampling_time, 3),
        "sampler":           sampler_desc,
        "algorithm":         "quantum_annealing_qubo",
        "paper_reference":   "Guth Jarkovsky et al. 2026 — adapted for Indian Railway Network",
    }
    summary_path = os.path.join(OUT_DIR, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)

    # QUBO stats standalone JSON
    qubo_path = os.path.join(OUT_DIR, "qubo_statistics.json")
    with open(qubo_path, "w") as f:
        json.dump(qubo_stats, f, indent=2)

    # Plots
    energy_plot = os.path.join(OUT_DIR, "qa_energy_landscape.png")
    plot_qubo_energy_landscape(sample_history, energy_plot, sampler_name)

    histogram_plot = os.path.join(OUT_DIR, "qa_energy_histogram.png")
    plot_energy_histogram(sample_history, histogram_plot, sampler_name)

    # QUBO structure plot (skipped automatically if n > 500)
    bqm_plot, var_index_plot, _, _ = build_qubo(
        trips_list, fleet_list, dist_lookup, maint_lookup
    )
    structure_plot = os.path.join(OUT_DIR, "qubo_matrix_structure.png")
    plot_qubo_structure(bqm_plot, var_index_plot,
                        trips_list, fleet_list, structure_plot)
    del bqm_plot

    # Cost history (best cost per sample)
    cost_plot = os.path.join(OUT_DIR, "cost_history.png")
    plot_cost_history(
        [{"step": r["step"], "total_cost": r["best_cost"],
          "best_cost": r["best_cost"]} for r in sample_history],
        f"Quantum Annealing ({sampler_name})",
        cost_plot
    )

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