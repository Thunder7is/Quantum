# NSGA-II multi-objective evolutionary algorithm solver for rolling stock scheduling
import os
import sys
import json
import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Support execution as direct script or package module
try:
    from ..utils.data_loader import load_instance, get_lookups, travel
    from ..utils.cost import compute_cost
    from ..utils.feasibility import repair_chromosome
    from ..utils.greedy_init import greedy_construction, chains_to_schedule_df
    from ..utils.viz import plot_cost_history
except (ImportError, ValueError):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.utils.data_loader import load_instance, get_lookups, travel
    from src.utils.cost import compute_cost
    from src.utils.feasibility import repair_chromosome
    from src.utils.greedy_init import greedy_construction, chains_to_schedule_df
    from src.utils.viz import plot_cost_history

# NSGA-II hyperparameters matching research specification
POPULATION_SIZE = 60
GENERATIONS = 200
ETA_C = 15.0
CROSSOVER_PROB = 0.9
ETA_M = 20.0
SEED = 42
MAX_COUPLE = 2

# Output directory path for NSGA-II artifacts
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "nsga2"))

# Map discrete 60-trip chromosome into rolling stock unit circulation chains
def chromosome_to_chains(chromosome: list, fleet_list: list, trips_list: list, type_to_unit_indices: dict) -> dict:
    chains = {u["unit_id"]: [] for u in fleet_list}
    for trip_idx, u_idx in enumerate(chromosome):
        trip = trips_list[trip_idx]
        primary_unit = fleet_list[u_idx]
        assigned_uids = [primary_unit["unit_id"]]

        # Coupling: add second unit from same type pool if passenger demand exceeds single capacity
        if trip.get("expected_passenger_demand", 0) > primary_unit["capacity"]:
            candidates = [idx for idx in type_to_unit_indices[primary_unit["unit_type"]] if idx != u_idx]
            if candidates:
                sec_idx = candidates[(trip_idx + u_idx) % len(candidates)]
                sec_uid = fleet_list[sec_idx]["unit_id"]
                if len(assigned_uids) < MAX_COUPLE and sec_uid not in assigned_uids:
                    assigned_uids.append(sec_uid)

        for uid in assigned_uids:
            chains[uid].append(trip["trip_id"])

    return chains

# Evaluate chromosome across the 3 simultaneous objectives
def evaluate_chromosome(chromosome: list, fleet_list: list, trips_list: list, type_to_unit_indices: dict,
                        trips_lookup: dict, fleet_lookup: dict, dist_lookup: dict, maint_lookup: dict):
    chains = chromosome_to_chains(chromosome, fleet_list, trips_list, type_to_unit_indices)
    bd = compute_cost(chains, trips_lookup, fleet_lookup, dist_lookup, maint_lookup)
    # Objective 1: Deadhead transit kilometers
    f1 = float(bd.get("deadhead_km", bd.get("total_deadhead_km", 0.0)))
    # Objective 2: Number of distinct trainset units activated
    f2 = float(bd.get("units_used", 0))
    # Objective 3: Uncovered passenger demand + 100x timing violations constraint penalty
    uncovered = float(bd.get("uncovered_demand", bd.get("uncovered_demand_passengers", 0)))
    timing = float(bd.get("timing_violations", 0))
    f3 = uncovered + timing * 100.0
    return (f1, f2, f3), bd["total"], bd, chains

# Pareto dominance check: returns True if objective vector p dominates vector q
def dominates(p: tuple, q: tuple) -> bool:
    return (p[0] <= q[0] and p[1] <= q[1] and p[2] <= q[2]) and (p[0] < q[0] or p[1] < q[1] or p[2] < q[2])

# Fast non-dominated sorting partitioning population into Pareto ranks
def fast_non_dominated_sort(objectives_list: list) -> list:
    n = len(objectives_list)
    domination_count = [0] * n
    dominated_solutions = [[] for _ in range(n)]
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if dominates(objectives_list[p], objectives_list[q]):
                dominated_solutions[p].append(q)
            elif dominates(objectives_list[q], objectives_list[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts

# Compute crowding distance for a specific non-dominated front
def compute_crowding_distance(front: list, objectives_list: list) -> dict:
    distance = {idx: 0.0 for idx in front}
    num_solutions = len(front)
    if num_solutions <= 2:
        for idx in front:
            distance[idx] = float("inf")
        return distance

    # Iterate over all three objective dimensions
    for m in range(3):
        sorted_front = sorted(front, key=lambda idx: objectives_list[idx][m])
        distance[sorted_front[0]] = float("inf")
        distance[sorted_front[-1]] = float("inf")
        denom = objectives_list[sorted_front[-1]][m] - objectives_list[sorted_front[0]][m]
        if denom == 0:
            continue
        for i in range(1, num_solutions - 1):
            distance[sorted_front[i]] += (objectives_list[sorted_front[i + 1]][m] - objectives_list[sorted_front[i - 1]][m]) / denom

    return distance

# Binary tournament selection based on rank and crowding distance
def binary_tournament(i1: int, i2: int, rank: dict, crowding: dict) -> int:
    if rank[i1] < rank[i2]:
        return i1
    if rank[i2] < rank[i1]:
        return i2
    # Within same rank, prefer solution with larger crowding distance
    if crowding[i1] > crowding[i2]:
        return i1
    return i2

# Simulated Binary Crossover (SBX) with distribution index eta=15
def sbx_crossover(parent1: list, parent2: list, num_units: int, rng: random.Random):
    if rng.random() > CROSSOVER_PROB:
        return list(parent1), list(parent2)

    length = len(parent1)
    child1 = list(parent1)
    child2 = list(parent2)

    for i in range(length):
        if rng.random() <= 0.5:
            p1_val = float(parent1[i])
            p2_val = float(parent2[i])
            if abs(p1_val - p2_val) > 1e-6:
                u = rng.random()
                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (ETA_C + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (ETA_C + 1.0))
                c1 = 0.5 * ((1.0 + beta) * p1_val + (1.0 - beta) * p2_val)
                c2 = 0.5 * ((1.0 - beta) * p1_val + (1.0 + beta) * p2_val)
                child1[i] = int(round(max(0, min(num_units - 1, c1))))
                child2[i] = int(round(max(0, min(num_units - 1, c2))))

    return child1, child2

# Polynomial mutation with distribution index eta=20 and prob=1/n_genes
def polynomial_mutation(chromosome: list, num_units: int, rng: random.Random) -> list:
    mutated = list(chromosome)
    length = len(mutated)
    mut_prob = 1.0 / length

    for i in range(length):
        if rng.random() < mut_prob:
            val = float(mutated[i])
            u = rng.random()
            if u <= 0.5:
                delta = (2.0 * u) ** (1.0 / (ETA_M + 1.0)) - 1.0
            else:
                delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (ETA_M + 1.0))
            new_val = val + delta * (num_units - 1)
            mutated[i] = int(round(max(0, min(num_units - 1, new_val))))

    return mutated

# Core NSGA-II optimization solver
def solve_nsga2(fleet, trips, dist_lookup, maint_lookup,
                pop_size=POPULATION_SIZE, generations=GENERATIONS, seed=SEED):
    rng = random.Random(seed)
    fleet_lookup, trips_lookup = get_lookups(fleet, trips)
    trips_sorted = trips.sort_values("departure_min").reset_index(drop=True)
    trips_list = trips_sorted.to_dict("records")
    fleet_list = fleet.to_dict("records")
    num_units = len(fleet_list)
    num_trips = len(trips_list)

    # Group unit indices by rolling stock type
    type_to_unit_indices = {}
    for idx, u in enumerate(fleet_list):
        type_to_unit_indices.setdefault(u["unit_type"], []).append(idx)

    # Step 1: Build seed_chromosome from greedy_construction()
    inst = {"fleet": fleet, "trips": trips_sorted, "dist_lookup": dist_lookup, "maint_lookup": maint_lookup}
    greedy_chains = greedy_construction(inst)
    unit_id_to_idx = {u["unit_id"]: idx for idx, u in enumerate(fleet_list)}
    trip_id_to_idx = {t["trip_id"]: idx for idx, t in enumerate(trips_list)}

    seed_chromosome = [0] * num_trips
    assigned_trips = set()
    for uid, chain in greedy_chains.items():
        u_idx = unit_id_to_idx[uid]
        for tid in chain:
            if tid in trip_id_to_idx:
                t_idx = trip_id_to_idx[tid]
                if t_idx not in assigned_trips:
                    seed_chromosome[t_idx] = u_idx
                    assigned_trips.add(t_idx)

    # Fallback assignment for any trips uncovered by greedy
    for t_idx in range(num_trips):
        if t_idx not in assigned_trips:
            seed_chromosome[t_idx] = t_idx % num_units

    # Initialize population[0] = seed_chromosome
    population = [list(seed_chromosome)]

    # population[1] to population[pop_size - 1] = seed_chromosome with 5-10 random gene swaps
    for _ in range(1, pop_size):
        chrom = list(seed_chromosome)
        num_mutations = rng.randint(5, 10)
        positions = rng.sample(range(num_trips), min(num_mutations, num_trips))
        for pos in positions:
            choices = [u for u in range(num_units) if u != chrom[pos]]
            if choices:
                chrom[pos] = rng.choice(choices)
        # Apply repair operator to ensure near-greedy chromosome remains feasible
        chrom = repair_chromosome(chrom, fleet_list, trips_list, dist_lookup, fleet_lookup)
        population.append(chrom)

    history_records = []

    # Generational evolution loop
    for gen in range(1, generations + 1):
        # Step 2: Fast non-dominated sort on current population
        pop_eval = [
            evaluate_chromosome(chrom, fleet_list, trips_list, type_to_unit_indices,
                                trips_lookup, fleet_lookup, dist_lookup, maint_lookup)
            for chrom in population
        ]
        objectives_list = [r[0] for r in pop_eval]
        fronts = fast_non_dominated_sort(objectives_list)

        # Assign rank to each individual
        rank = {}
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r

        # Step 3: Compute crowding distance for each front
        crowding = {}
        for front in fronts:
            cd_dict = compute_crowding_distance(front, objectives_list)
            crowding.update(cd_dict)

        # Log metrics for generation history
        best_f1 = min(obj[0] for obj in objectives_list)
        best_f2 = min(obj[1] for obj in objectives_list)
        best_f3 = min(obj[2] for obj in objectives_list)
        n_pareto = len(fronts[0]) if fronts else 0

        history_records.append({
            "generation": gen,
            "step": gen,
            "best_f1": round(best_f1, 1),
            "best_f2": int(best_f2),
            "best_f3": round(best_f3, 1),
            "n_pareto": n_pareto,
            "best_cost": round(best_f1, 1),
            "total_cost": round(best_f1, 1),
        })

        # Step 4 & 5: Selection, SBX crossover, polynomial mutation, and repair to create offspring
        offspring = []
        while len(offspring) < pop_size:
            p1_idx = binary_tournament(rng.randint(0, pop_size - 1), rng.randint(0, pop_size - 1), rank, crowding)
            p2_idx = binary_tournament(rng.randint(0, pop_size - 1), rng.randint(0, pop_size - 1), rank, crowding)

            c1, c2 = sbx_crossover(population[p1_idx], population[p2_idx], num_units, rng)
            c1 = polynomial_mutation(c1, num_units, rng)
            c2 = polynomial_mutation(c2, num_units, rng)

            # Repair offspring chromosomes
            c1 = repair_chromosome(c1, fleet_list, trips_list, dist_lookup, fleet_lookup)
            c2 = repair_chromosome(c2, fleet_list, trips_list, dist_lookup, fleet_lookup)

            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        # Step 6: Combine parent and offspring (2N population), re-evaluate, re-sort, and select top N
        combined_population = population + offspring
        combined_eval = [
            evaluate_chromosome(chrom, fleet_list, trips_list, type_to_unit_indices,
                                trips_lookup, fleet_lookup, dist_lookup, maint_lookup)
            for chrom in combined_population
        ]
        combined_objs = [r[0] for r in combined_eval]
        combined_fronts = fast_non_dominated_sort(combined_objs)

        next_population = []
        for front in combined_fronts:
            if len(next_population) + len(front) <= pop_size:
                next_population.extend([combined_population[i] for i in front])
            else:
                cd_front = compute_crowding_distance(front, combined_objs)
                sorted_front = sorted(front, key=lambda idx: cd_front[idx], reverse=True)
                needed = pop_size - len(next_population)
                next_population.extend([combined_population[i] for i in sorted_front[:needed]])
                break

        population = next_population

    # Final population evaluation on repaired chromosomes
    final_eval = [
        evaluate_chromosome(chrom, fleet_list, trips_list, type_to_unit_indices,
                            trips_lookup, fleet_lookup, dist_lookup, maint_lookup)
        for chrom in population
    ]
    final_objs = [r[0] for r in final_eval]
    final_fronts = fast_non_dominated_sort(final_objs)
    pareto_indices = final_fronts[0]

    # Compute final crowding distances for Pareto front
    final_crowding = compute_crowding_distance(pareto_indices, final_objs)

    # Build Pareto front records
    pareto_points = []
    for idx in pareto_indices:
        pareto_points.append({
            "deadhead_km": round(final_objs[idx][0], 1),
            "units_used": int(final_objs[idx][1]),
            "demand_penalty": round(final_objs[idx][2], 1),
            "rank": 0,
            "generation": generations,
            "crowding_distance": final_crowding[idx],
            "chromosome": population[idx],
            "chains": final_eval[idx][3],
            "breakdown": final_eval[idx][2],
            "scalarized_cost": final_eval[idx][1],
        })

    # Select best compromise schedule based on minimum scalarized cost
    best_compromise = min(pareto_points, key=lambda p: p["scalarized_cost"])
    best_chains = best_compromise["chains"]
    best_cost = best_compromise["scalarized_cost"]
    best_breakdown = best_compromise["breakdown"]

    return best_chains, best_cost, best_breakdown, pareto_points, history_records

# Main execution routine
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    inst = load_instance()
    fleet, trips, dist_lookup, maint_lookup = inst["fleet"], inst["trips"], inst["dist_lookup"], inst["maint_lookup"]

    print("=================================================================")
    print("  NSGA-II Multi-Objective Evolutionary Algorithm")
    print(f"  Fleet: {len(fleet)} units | Trips: {len(trips)} | Population: {POPULATION_SIZE}")
    print(f"  Generations: {GENERATIONS} | Objectives: (f1: deadhead, f2: units, f3: penalty)")
    print("=================================================================\n")

    # Run NSGA-II solver
    best_chains, best_cost, best_breakdown, pareto_points, history_records = solve_nsga2(
        fleet, trips, dist_lookup, maint_lookup
    )

    print("NSGA-II best compromise solution breakdown:")
    print(json.dumps(best_breakdown, indent=2))
    print(f"\nFinal Pareto optimal solutions count: {len(pareto_points)}")

    # Save final schedule CSV
    schedule_df = chains_to_schedule_df(best_chains, trips)
    schedule_path = os.path.join(OUT_DIR, "final_schedule.csv")
    schedule_df.to_csv(schedule_path, index=False)

    # Save pareto_front.csv with columns: deadhead_km, units_used, demand_penalty, rank, generation
    pareto_df = pd.DataFrame(pareto_points)[["deadhead_km", "units_used", "demand_penalty", "rank", "generation"]]
    pareto_csv_path = os.path.join(OUT_DIR, "pareto_front.csv")
    pareto_df.to_csv(pareto_csv_path, index=False)

    # Save run_summary.json with Pareto stats
    f1_vals = [p["deadhead_km"] for p in pareto_points]
    f2_vals = [p["units_used"] for p in pareto_points]
    f3_vals = [p["demand_penalty"] for p in pareto_points]
    summary_path = os.path.join(OUT_DIR, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "n_pareto_solutions": len(pareto_points),
            "objectives_summary": {
                "f1_deadhead_km": {"min": min(f1_vals), "max": max(f1_vals)},
                "f2_units_used": {"min": min(f2_vals), "max": max(f2_vals)},
                "f3_demand_penalty": {"min": min(f3_vals), "max": max(f3_vals)},
            },
            "best_compromise_breakdown": best_breakdown,
            "algorithm": "nsga2",
            "generations": GENERATIONS,
            "population_size": POPULATION_SIZE,
        }, f, indent=2)

    # Save cost_history.csv with generation, best_f1, best_f2, best_f3, n_pareto
    cost_hist_df = pd.DataFrame(history_records)[["generation", "best_f1", "best_f2", "best_f3", "n_pareto"]]
    cost_hist_csv = os.path.join(OUT_DIR, "cost_history.csv")
    cost_hist_df.to_csv(cost_hist_csv, index=False)

    # Save cost_history.png via plot_cost_history() using best_f1 per generation
    plot_cost_path = os.path.join(OUT_DIR, "cost_history.png")
    plot_cost_history(history_records, "NSGA-II (Best Deadhead km)", plot_cost_path)

    # Save pareto_front.png: 3D scatter of f1 vs f2 vs f3 colored by crowding distance
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    cd_raw = [p["crowding_distance"] for p in pareto_points]
    finite_cds = [c for c in cd_raw if not np.isinf(c)]
    max_c = (max(finite_cds) * 1.5) if finite_cds else 1.0
    cd_colors = [c if not np.isinf(c) else max_c for c in cd_raw]

    scatter = ax.scatter(f1_vals, f2_vals, f3_vals, c=cd_colors, cmap="viridis", s=50, edgecolors="k")
    ax.set_xlabel("Deadhead Distance (km) [f1]", fontsize=10, labelpad=8)
    ax.set_ylabel("Units Used [f2]", fontsize=10, labelpad=8)
    ax.set_zlabel("Demand & Timing Penalty [f3]", fontsize=10, labelpad=8)
    ax.set_title("NSGA-II Final Pareto Optimal Front (3D)", fontsize=12, fontweight="bold")
    fig.colorbar(scatter, ax=ax, label="Crowding Distance", pad=0.1)
    plt.tight_layout()
    pareto_img_path = os.path.join(OUT_DIR, "pareto_front.png")
    plt.savefig(pareto_img_path, dpi=300)
    plt.close(fig)

    print(f"Saved schedule: {schedule_path}")
    print(f"Saved Pareto CSV: {pareto_csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved cost history CSV: {cost_hist_csv}")
    print(f"Saved cost history plot: {plot_cost_path}")
    print(f"Saved Pareto 3D plot: {pareto_img_path}")

# Standalone execution entrypoint
if __name__ == "__main__":
    main()
