# Quantum Annealing with Divide-and-Conquer for rolling stock scheduling
# Approach: Guth Jarkovsky et al. (2026) - hybrid D&C with quantum subproblem solver
# Each iteration solves a small subgraph of the full QUBO using TabuSampler
# This allows quantum-quality solutions on problems too large for direct QPU submission
import os
import sys
import json
import time
import itertools
import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Sampler imports ───────────────────────────────────────────────────────────
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
    from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler as DimodSA
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

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "quantum_dc")
)

# ─── D&C Hyperparameters ─────────────────────────────────────────────────────
# Subgraph size: number of trip-nodes in each MWIS subgraph sent to the solver
# Paper uses 20 for QPU (hardware limit), we can go higher classically
SUBGRAPH_SIZE          = 30    # trips per subgraph (small = fast, large = better quality)
SUBGRAPH_SIZE_LARGE    = 35    # used after MODE_SWITCH_THRESHOLD unserved trips remain
MODE_SWITCH_THRESHOLD  = 20    # switch to larger subgraph when trips remaining <= this
EMPTY_TRIP_THRESHOLD   = 9     # allow depot-hop repositioning trips when remaining <= this
MAX_OUTER_ITERATIONS   = 60    # safety cap on outer D&C loop
RANDOM_SEED            = 42

# ─── TabuSampler config per subgraph call ────────────────────────────────────
# Each subgraph is tiny (10-20 nodes) so reads can be high quality and fast
TABU_READS_PER_SUBGRAPH   = 1000   # independent restarts per subgraph
TABU_TENURE_SUBGRAPH      = 7    # tabu tenure (smaller for small graphs)
TABU_TIMEOUT_SUBGRAPH     = 90    # seconds per subgraph solve

# Neal fallback config
NEAL_READS_SUBGRAPH       = 700
NEAL_SWEEPS_SUBGRAPH      = 2500


# ─── STEP 1: Build MWIS graph ────────────────────────────────────────────────

def build_mwis_graph(trips_list: list, fleet_list: list,
                     dist_lookup: dict, maint_lookup: dict) -> tuple:
    """
    Map rolling stock scheduling to Maximum Weight Independent Set (MWIS).

    Following Guth Jarkovsky et al. (2026):
    - NODE: each feasible (unit, trip) assignment = one node
    - WEIGHT: reward for covering the trip (passenger demand proxy)
              minus penalty for deadhead km
              w_{u,t} = 2 * trip_km - deadhead_km(u -> trip_origin)
              (coefficient 2 prioritises serving trips over pure cost reduction)
    - EDGE: connect two nodes if they are mutually incompatible:
            (u1, t) -- (u2, t): two units both claiming the same trip
            (u, t1) -- (u, t2): same unit assigned to two trips it cannot chain

    Returns:
        nodes: list of (unit_id, trip_id, weight)
        edges: set of (node_idx_a, node_idx_b) incompatible pairs
        node_index: (unit_id, trip_id) -> node_idx
    """
    # Build feasible arc lookup: unit_id -> set of (trip_i_id, trip_j_id)
    feasible_arcs = {}
    for unit in fleet_list:
        uid = unit["unit_id"]
        feasible_arcs[uid] = set()
        for t_a in trips_list:
            for t_b in trips_list:
                if t_a["trip_id"] == t_b["trip_id"]:
                    continue
                if t_a["departure_min"] >= t_b["departure_min"]:
                    continue
                tt, _ = travel(dist_lookup,
                               t_a["destination_station"],
                               t_b["origin_station"])
                if t_a["arrival_min"] + t_a["min_turnaround_min"] + tt <= t_b["departure_min"]:
                    feasible_arcs[uid].add((t_a["trip_id"], t_b["trip_id"]))

    # Build maintenance km tracker per unit
    maint_max = {u["unit_id"]: maint_lookup.get(u["unit_type"], 12000)
                 for u in fleet_list}

    # Build nodes: only include (u, t) pairs where unit can FIRST reach trip
    nodes = []
    node_index = {}
    unit_loc = {u["unit_id"]: u["current_location"] for u in fleet_list}
    unit_avail = {u["unit_id"]: u["available_from_min"] for u in fleet_list}
    unit_km = {u["unit_id"]: u["km_since_maintenance"] for u in fleet_list}

    for unit in fleet_list:
        uid = unit["unit_id"]
        for trip in trips_list:
            tid = trip["trip_id"]
            # Check unit can reach trip origin in time
            tt, dh_km = travel(dist_lookup, unit_loc[uid], trip["origin_station"])
            if unit_avail[uid] + tt > trip["departure_min"]:
                continue
            # Check maintenance limit
            if unit_km[uid] + dh_km + trip["distance_km"] > maint_max[uid]:
                continue
            # Node weight: 2 * full_km - deadhead_km (paper formula)
            weight = 2.0 * trip["distance_km"] - dh_km
            idx = len(nodes)
            nodes.append((uid, tid, weight))
            node_index[(uid, tid)] = idx

    # Build edges: incompatibility constraints
    edges = set()
    for i, (uid_a, tid_a, _) in enumerate(nodes):
        for j, (uid_b, tid_b, _) in enumerate(nodes):
            if j <= i:
                continue
            # Same trip served by two different units (conflict)
            if tid_a == tid_b and uid_a != uid_b:
                edges.add((i, j))
                continue
            # Same unit assigned to two trips it cannot chain (both directions)
            if uid_a == uid_b and tid_a != tid_b:
                t_a = next(t for t in trips_list if t["trip_id"] == tid_a)
                t_b = next(t for t in trips_list if t["trip_id"] == tid_b)
                if t_a["departure_min"] < t_b["departure_min"]:
                    if (tid_a, tid_b) not in feasible_arcs.get(uid_a, set()):
                        edges.add((i, j))
                else:
                    if (tid_b, tid_a) not in feasible_arcs.get(uid_a, set()):
                        edges.add((i, j))

    return nodes, edges, node_index, feasible_arcs


# ─── STEP 2: Select subgraph ─────────────────────────────────────────────────

def select_subgraph(nodes: list, edges: set, unserved_trip_ids: set,
                    subgraph_size: int, rng: random.Random) -> list:
    """
    Select a subgraph of size k from the remaining MWIS graph.

    Strategy from paper: prefer nodes (unit, trip) pairs that cover
    the most unserved trips AND have the highest weight.
    This biases toward near-optimal cycles while maintaining diversity.

    Returns: list of node indices in the subgraph.
    """
    # Filter to only nodes involving unserved trips
    candidates = [i for i, (uid, tid, w) in enumerate(nodes)
                  if tid in unserved_trip_ids]

    if not candidates:
        return []

    # Score each candidate: weight + bonus for unserved trip coverage
    def score(idx):
        uid, tid, w = nodes[idx]
        served_bonus = 500.0 if tid in unserved_trip_ids else 0.0
        return w + served_bonus

    # Sort by score descending, add a small random perturbation for diversity
    scored = sorted(candidates,
                    key=lambda i: score(i) + rng.uniform(0, 100),
                    reverse=True)

    # Take top subgraph_size nodes
    return scored[:subgraph_size]


# ─── STEP 3: Build QUBO for subgraph ─────────────────────────────────────────

def build_subgraph_qubo(subgraph_node_indices: list, nodes: list,
                        edges: set) -> tuple:
    """
    Build a small QUBO BQM for just the selected subgraph nodes.

    MWIS QUBO formulation (standard):
        maximise  sum_i w_i * x_i
        subject to  x_i * x_j = 0  for all edges (i,j)

    Converting to minimisation for annealer:
        minimise  -sum_i w_i * x_i
                  + LAMBDA * sum_{(i,j) in E} x_i * x_j

    Returns: (bqm, local_index_map)
    local_index_map: global_node_idx -> local_bqm_variable_idx
    """
    if not DIMOD_AVAILABLE:
        raise ImportError("dimod required. Run: pip install dimod")

    LAMBDA_MWIS = 5000.0   # must be > max node weight to enforce independence

    local_map = {global_idx: local_idx
                 for local_idx, global_idx in enumerate(subgraph_node_indices)}

    bqm = BinaryQuadraticModel(vartype=dimod.BINARY)

    # Linear terms: -weight (we minimise so negate the maximisation objective)
    for global_idx in subgraph_node_indices:
        _, _, weight = nodes[global_idx]
        local_idx = local_map[global_idx]
        bqm.add_variable(local_idx, -weight)

    # Quadratic terms: penalty for incompatible pairs within subgraph
    for (i, j) in edges:
        if i in local_map and j in local_map:
            bqm.add_interaction(local_map[i], local_map[j], LAMBDA_MWIS)

    return bqm, local_map


# ─── STEP 4: Solve subgraph QUBO ─────────────────────────────────────────────

def solve_subgraph(bqm, num_reads: int, timeout: int) -> dict:
    """
    Solve the subgraph QUBO using best available sampler.
    Returns the best sample as {var_idx: 0/1}.
    """
    if TABU_AVAILABLE:
        sampler = TabuSampler()
        # TabuSampler requires 0 <= tenure <= num_vars - 1
        num_vars = len(bqm.variables)
        tenure = min(TABU_TENURE_SUBGRAPH, max(0, num_vars - 1))

        response = sampler.sample(
            bqm,
            num_reads=num_reads,
            tenure=TABU_TENURE_SUBGRAPH,
            timeout=timeout,
        )
    elif NEAL_AVAILABLE:
        sampler = neal.SimulatedAnnealingSampler()
        response = sampler.sample(
            bqm,
            num_reads=NEAL_READS_SUBGRAPH,
            num_sweeps=NEAL_SWEEPS_SUBGRAPH,
            seed=RANDOM_SEED,
        )
    else:
        sampler = DimodSA()
        response = sampler.sample(
            bqm,
            num_reads=100,
            num_sweeps=500,
            seed=RANDOM_SEED,
        )

    # Return best sample (lowest energy = best MWIS solution)
    return dict(response.first.sample)


# ─── STEP 5: Pruning ─────────────────────────────────────────────────────────

def prune_solution(sample: dict, subgraph_node_indices: list,
                   local_map: dict, nodes: list, edges: set) -> list:
    """
    Convert a raw QUBO sample to a valid independent set.
    Pruning removes conflicting nodes to ensure no two selected nodes share an edge.

    Algorithm (from paper):
    1. Find all selected nodes (value=1) that have edges between them.
    2. Assign each such node a ratio = weight / degree_in_subgraph.
    3. Remove node with lowest ratio (worst weight per conflict).
    4. Repeat until no conflicting edges remain.

    Returns: list of global node indices in the pruned independent set.
    """
    # Start with all selected nodes
    reverse_local = {v: k for k, v in local_map.items()}
    selected = set(reverse_local[local_idx]
                   for local_idx, val in sample.items()
                   if val == 1 and local_idx in reverse_local)

    # Iteratively remove lowest-ratio conflicting node
    while True:
        # Find conflicting edges within selected set
        conflicts = [(i, j) for (i, j) in edges
                     if i in selected and j in selected]
        if not conflicts:
            break

        # Build degree count for conflicting nodes
        degree = {}
        for (i, j) in conflicts:
            degree[i] = degree.get(i, 0) + 1
            degree[j] = degree.get(j, 0) + 1

        # Remove node with lowest weight-to-degree ratio
        worst = min(degree.keys(),
                    key=lambda idx: nodes[idx][2] / degree[idx])
        selected.discard(worst)

    return list(selected)


# ─── STEP 6: Update global graph ─────────────────────────────────────────────

def update_graph(selected_nodes: list, nodes: list, edges: set,
                 unserved_trip_ids: set, solution_chains: dict) -> tuple:
    """
    After solving a subgraph:
    1. Add selected (unit, trip) assignments to solution_chains.
    2. Remove selected nodes and all their neighbours from the graph.
       (Neighbours = nodes sharing an edge with any selected node)
    3. Remove covered trips from unserved_trip_ids.

    Returns: (updated_nodes, updated_edges, updated_unserved_trip_ids,
              updated_solution_chains, n_newly_served)
    """
    if not selected_nodes:
        return nodes, edges, unserved_trip_ids, solution_chains, 0

    # Collect all nodes to remove: selected + their neighbours
    to_remove = set(selected_nodes)
    for sel_idx in selected_nodes:
        for (i, j) in edges:
            if i == sel_idx:
                to_remove.add(j)
            elif j == sel_idx:
                to_remove.add(i)

    # Record assignments from selected nodes
    n_newly_served = 0
    for idx in selected_nodes:
        uid, tid, _ = nodes[idx]
        solution_chains[uid].append(tid)
        if tid in unserved_trip_ids:
            unserved_trip_ids = unserved_trip_ids - {tid}
            n_newly_served += 1

    # Rebuild nodes and edges without removed indices
    old_to_new = {}
    new_nodes = []
    for old_idx, node in enumerate(nodes):
        if old_idx not in to_remove:
            new_idx = len(new_nodes)
            old_to_new[old_idx] = new_idx
            new_nodes.append(node)

    new_edges = set()
    for (i, j) in edges:
        if i in old_to_new and j in old_to_new:
            new_edges.add((old_to_new[i], old_to_new[j]))

    return new_nodes, new_edges, unserved_trip_ids, solution_chains, n_newly_served


# ─── STEP 7: Main D&C Solver ─────────────────────────────────────────────────

def solve_quantum_dc(fleet, trips, dist_lookup, maint_lookup,
                     subgraph_size: int = SUBGRAPH_SIZE,
                     seed: int = RANDOM_SEED) -> tuple:
    """
    Divide-and-Conquer Quantum Annealing solver.

    Algorithm (following Guth Jarkovsky et al. 2026):
    ─────────────────────────────────────────────────
    1. Build the full MWIS graph (nodes = feasible unit-trip pairs,
       edges = incompatible pairs).
    2. OUTER LOOP (until all trips served or max iterations reached):
       a. Select a subgraph of size k from unserved trip nodes.
       b. Build QUBO for that subgraph.
       c. Solve QUBO with TabuSampler.
       d. Prune result to valid independent set.
       e. Add selected assignments to solution; remove from graph.
    3. If trips still unserved after main loop, run a greedy fallback
       to assign remaining trips.
    4. Evaluate full solution with compute_cost().

    Returns: (chains, best_cost, breakdown, iteration_log, total_time)
    """
    if not DIMOD_AVAILABLE:
        raise ImportError("dimod required. Run: pip install dimod")

    rng = random.Random(seed)
    trips_sorted = trips.sort_values("departure_min").reset_index(drop=True)
    trips_list   = trips_sorted.to_dict("records")
    fleet_list   = fleet.to_dict("records")
    fleet_lookup, trips_lookup = get_lookups(fleet, trips_sorted)

    all_trip_ids    = {t["trip_id"] for t in trips_list}
    unserved        = set(all_trip_ids)
    solution_chains = {u["unit_id"]: [] for u in fleet_list}

    print("Building MWIS graph...")
    t0 = time.time()
    nodes, edges, node_index, feasible_arcs = build_mwis_graph(
        trips_list, fleet_list, dist_lookup, maint_lookup
    )
    print(f"MWIS graph: {len(nodes)} nodes | {len(edges)} edges | built in {time.time()-t0:.2f}s")

    sampler_name = ("TabuSampler" if TABU_AVAILABLE
                    else "NealSampler" if NEAL_AVAILABLE
                    else "DimodSASampler")

    iteration_log = []
    outer_iter    = 0
    total_subgraph_time = 0.0
    t_total_start = time.time()

    print(f"\nStarting D&C outer loop | Sampler: {sampler_name}")
    print(f"Subgraph size: {subgraph_size} | Mode-switch at {MODE_SWITCH_THRESHOLD} remaining\n")

    while unserved and outer_iter < MAX_OUTER_ITERATIONS:
        outer_iter += 1
        n_remaining = len(unserved)

        # Adaptive subgraph size: switch to larger subgraph near end
        k = SUBGRAPH_SIZE_LARGE if n_remaining <= MODE_SWITCH_THRESHOLD else subgraph_size

        print(f"[Iter {outer_iter:02d}] Unserved: {n_remaining:2d} trips | "
              f"Subgraph size: {k} | Graph: {len(nodes)} nodes")

        # 2a: select subgraph
        subgraph_indices = select_subgraph(nodes, edges, unserved, k, rng)

        if not subgraph_indices:
            print(f"  No valid subgraph found. Breaking outer loop.")
            break

        # 2b: build subgraph QUBO
        sub_bqm, local_map = build_subgraph_qubo(subgraph_indices, nodes, edges)

        # 2c: solve subgraph QUBO
        t_sub = time.time()
        raw_sample = solve_subgraph(
            sub_bqm,
            num_reads=TABU_READS_PER_SUBGRAPH,
            timeout=TABU_TIMEOUT_SUBGRAPH,
        )
        sub_time = time.time() - t_sub
        total_subgraph_time += sub_time

        # 2d: prune to valid independent set
        selected = prune_solution(raw_sample, subgraph_indices,
                                  local_map, nodes, edges)

        # 2e: update graph and solution
        nodes, edges, unserved, solution_chains, n_served = update_graph(
            selected, nodes, edges, unserved, solution_chains
        )

        log_entry = {
            "iteration":         outer_iter,
            "unserved_before":   n_remaining,
            "unserved_after":    len(unserved),
            "trips_served":      n_served,
            "subgraph_size":     len(subgraph_indices),
            "selected_nodes":    len(selected),
            "subgraph_time_sec": round(sub_time, 3),
        }
        iteration_log.append(log_entry)

        print(f"  Served {n_served} trips in {sub_time:.1f}s | "
              f"Remaining: {len(unserved)}")

        # Safety: if we are not making progress, increase subgraph size
        if n_served == 0:
            subgraph_size = min(subgraph_size + 5, 25)
            print(f"  No progress — increasing subgraph size to {subgraph_size}")

    # ── Greedy fallback for any trips still unserved ──────────────────────────
    n_fallback = 0
    if unserved:
        print(f"\nGreedy fallback for {len(unserved)} unserved trips...")
        # Build simple availability tracker
        unit_avail = {u["unit_id"]: u["available_from_min"] for u in fleet_list}
        unit_loc   = {u["unit_id"]: u["current_location"]   for u in fleet_list}
        unit_km    = {u["unit_id"]: u["km_since_maintenance"] for u in fleet_list}
        maint_max  = {u["unit_id"]: maint_lookup.get(u["unit_type"], 12000)
                      for u in fleet_list}

        # Update state from already-assigned trips
        for uid, trip_chain in solution_chains.items():
            for tid in sorted(trip_chain,
                              key=lambda x: trips_lookup[x]["departure_min"]):
                t = trips_lookup[tid]
                _, dh = travel(dist_lookup, unit_loc[uid], t["origin_station"])
                unit_km[uid]    += dh + t["distance_km"]
                unit_avail[uid]  = t["arrival_min"] + t["min_turnaround_min"]
                unit_loc[uid]    = t["destination_station"]

        for tid in sorted(unserved,
                          key=lambda x: trips_lookup[x]["departure_min"]):
            trip = trips_lookup[tid]
            best_uid, best_arr = None, float("inf")
            for uid in unit_avail:
                tt, dh = travel(dist_lookup, unit_loc[uid], trip["origin_station"])
                arr = unit_avail[uid] + tt
                proj_km = unit_km[uid] + dh + trip["distance_km"]
                if proj_km > maint_max[uid]:
                    continue
                if arr < best_arr:
                    best_arr, best_uid = arr, uid
            if best_uid:
                solution_chains[best_uid].append(tid)
                _, dh = travel(dist_lookup, unit_loc[best_uid], trip["origin_station"])
                unit_km[best_uid]    += dh + trip["distance_km"]
                unit_avail[best_uid]  = trip["arrival_min"] + trip["min_turnaround_min"]
                unit_loc[best_uid]    = trip["destination_station"]
                n_fallback += 1

    total_time = time.time() - t_total_start

    # ── Evaluate final solution ───────────────────────────────────────────────
    breakdown = compute_cost(solution_chains, trips_lookup, fleet_lookup,
                             dist_lookup, maint_lookup)

    print(f"\nD&C complete | {outer_iter} iterations | "
          f"Fallback trips: {n_fallback} | Total time: {total_time:.1f}s")

    return solution_chains, breakdown["total"], breakdown, iteration_log, total_time


# ─── STEP 8: Visualisations ───────────────────────────────────────────────────

def plot_dc_convergence(iteration_log: list, out_path: str):
    """
    Two-panel chart showing the D&C algorithm's progress:
    Top: trips remaining unserved after each iteration.
    Bottom: time spent on each subgraph solve.
    """
    iters        = [r["iteration"]         for r in iteration_log]
    unserved     = [r["unserved_after"]    for r in iteration_log]
    times        = [r["subgraph_time_sec"] for r in iteration_log]
    trips_served = [r["trips_served"]      for r in iteration_log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Unserved trips decreasing
    ax1.step(iters, unserved, where="post", color="#9b59b6",
             linewidth=2.2, label="Unserved Trips Remaining")
    ax1.fill_between(iters, unserved, step="post",
                     alpha=0.15, color="#9b59b6")
    ax1.set_xlabel("D&C Iteration")
    ax1.set_ylabel("Unserved Trips")
    ax1.set_title("Divide-and-Conquer QA — Trip Coverage Progress")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Subgraph solve times
    ax2.bar(iters, times, color="#3498db", alpha=0.8, edgecolor="black")
    ax2.set_xlabel("D&C Iteration")
    ax2.set_ylabel("Subgraph Solve Time (s)")
    ax2.set_title("Time per Subgraph QUBO Solve")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dc_vs_full(dc_breakdown: dict, full_qa_breakdown: dict,
                    sa_breakdown: dict, greedy_breakdown: dict,
                    out_path: str):
    """
    Bar chart comparing D&C-QA against full QUBO QA, SA, and Greedy.
    """
    algos = ["Greedy", "SA", "Full QUBO\n(TabuSampler)", "D&C QA\n(TabuSampler)"]
    colors = ["#d62728", "#ff7f0e", "#9b59b6", "#2ecc71"]

    metrics = {
        "Total Cost":       [greedy_breakdown["total"],      sa_breakdown["total"],
                             full_qa_breakdown["total"],     dc_breakdown["total"]],
        "Deadhead km":      [greedy_breakdown["deadhead_km"],sa_breakdown["deadhead_km"],
                             full_qa_breakdown["deadhead_km"],dc_breakdown["deadhead_km"]],
        "Uncovered Demand": [greedy_breakdown["uncovered_demand"],
                             sa_breakdown["uncovered_demand"],
                             full_qa_breakdown["uncovered_demand"],
                             dc_breakdown["uncovered_demand"]],
        "Timing Violations":[greedy_breakdown["timing_violations"],
                             sa_breakdown["timing_violations"],
                             full_qa_breakdown["timing_violations"],
                             dc_breakdown["timing_violations"]],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_list = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]

    for (title, vals), ax in zip(metrics.items(), ax_list):
        bars = ax.bar(algos, vals, color=colors, width=0.5,
                      edgecolor="#222222", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{int(h):,}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", labelsize=9)

    plt.suptitle(
        "Algorithm Comparison: Greedy | SA | Full QUBO QA | D&C QA",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── STEP 9: Main Execution ───────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not DIMOD_AVAILABLE:
        print("ERROR: dimod required. Run: pip install dimod")
        return

    sampler_name = ("TabuSampler" if TABU_AVAILABLE
                    else "NealSampler" if NEAL_AVAILABLE
                    else "DimodSASampler")

    inst = load_instance()
    fleet        = inst["fleet"]
    trips        = inst["trips"]
    dist_lookup  = inst["dist_lookup"]
    maint_lookup = inst["maint_lookup"]
    fleet_lookup, trips_lookup = get_lookups(fleet, trips)

    print("=================================================================")
    print("  Quantum Annealing — Divide and Conquer (D&C QUBO)")
    print(f"  Fleet: {len(fleet)} units | Trips: {len(trips)} | Stations: 5")
    print(f"  Sampler: {sampler_name}")
    print(f"  Subgraph size: {SUBGRAPH_SIZE} (→ {SUBGRAPH_SIZE_LARGE} "
          f"after {MODE_SWITCH_THRESHOLD} trips remain)")
    print(f"  Max outer iterations: {MAX_OUTER_ITERATIONS}")
    print("=================================================================\n")

    chains, best_cost, breakdown, iteration_log, total_time = solve_quantum_dc(
        fleet, trips, dist_lookup, maint_lookup,
        subgraph_size=SUBGRAPH_SIZE, seed=RANDOM_SEED
    )

    print("\nD&C Quantum Annealing solution breakdown:")
    print(json.dumps(breakdown, indent=2))

    # ── Load comparison results from other algorithms ─────────────────────────
    base_out = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
    )

    def _load_bd(folder, key="breakdown"):
        path = os.path.join(base_out, folder, "run_summary.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            d = json.load(f)
        return d.get(key, d.get("breakdown", d.get("final_breakdown", {})))

    greedy_bd  = _load_bd("greedy")
    sa_bd      = _load_bd("sa", key="final_breakdown")
    full_qa_bd = _load_bd("quantum")

    # ── Save outputs ──────────────────────────────────────────────────────────

    # Final schedule
    schedule_df = chains_to_schedule_df(chains, trips)
    schedule_df.to_csv(os.path.join(OUT_DIR, "final_schedule.csv"), index=False)

    # Iteration log CSV
    pd.DataFrame(iteration_log).to_csv(
        os.path.join(OUT_DIR, "dc_iteration_log.csv"), index=False
    )

    # Run summary JSON
    run_summary = {
        "breakdown": breakdown,
        "total_time_sec": round(total_time, 2),
        "n_outer_iterations": len(iteration_log),
        "sampler": sampler_name,
        "hyperparameters": {
            "subgraph_size":           SUBGRAPH_SIZE,
            "subgraph_size_large":     SUBGRAPH_SIZE_LARGE,
            "mode_switch_threshold":   MODE_SWITCH_THRESHOLD,
            "tabu_reads_per_subgraph": TABU_READS_PER_SUBGRAPH,
            "tabu_tenure":             TABU_TENURE_SUBGRAPH,
            "tabu_timeout_per_sub":    TABU_TIMEOUT_SUBGRAPH,
            "lambda_mwis":             5000.0,
            "node_weight_formula":     "2 * trip_km - deadhead_km",
        },
        "algorithm":       "quantum_annealing_divide_and_conquer",
        "paper_reference": "Guth Jarkovsky et al. 2026 — adapted for Indian Railway Network",
    }
    with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)

    # D&C convergence chart
    plot_dc_convergence(
        iteration_log,
        os.path.join(OUT_DIR, "dc_convergence.png")
    )

    # Comparison chart (if other results exist)
    if greedy_bd and sa_bd and full_qa_bd:
        plot_dc_vs_full(
            breakdown, full_qa_bd, sa_bd, greedy_bd,
            os.path.join(OUT_DIR, "dc_vs_all_comparison.png")
        )
        print(f"Saved comparison: {os.path.join(OUT_DIR, 'dc_vs_all_comparison.png')}")

    print(f"\nSaved schedule:        {os.path.join(OUT_DIR, 'final_schedule.csv')}")
    print(f"Saved iteration log:   {os.path.join(OUT_DIR, 'dc_iteration_log.csv')}")
    print(f"Saved run summary:     {os.path.join(OUT_DIR, 'run_summary.json')}")
    print(f"Saved D&C chart:       {os.path.join(OUT_DIR, 'dc_convergence.png')}")
    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()