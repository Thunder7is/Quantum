# NSGA-II multi-objective evolutionary algorithm solver for rolling stock circulation
import os
import json
import random
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..utils.data_loader import load_instance, get_lookups
from ..utils.cost import compute_cost
from ..utils.greedy_init import greedy_construction, chains_to_schedule_df
from ..utils.viz import plot_cost_history

# Default NSGA-II parameters
POPULATION_SIZE = 30
GENERATIONS = 50
SEED = 42

# Directory path for NSGA-II outputs
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "nsga2"))

# Copy chains structure
def copy_chains(chains):
    return {uid: list(trips) for uid, trips in chains.items()}

# Evaluate two distinct objective vectors for multi-objective optimization
def evaluate_objectives(chains, fleet_lookup, trips_lookup, dist_lookup, maint_lookup):
    cost, bd = compute_cost(chains, fleet_lookup, trips_lookup, dist_lookup, maint_lookup)
    # Objective 1: Operational resource consumption (deadhead km + unit activation penalty)
    f1 = bd["total_deadhead_km"] + 150.0 * bd["units_used"]
    # Objective 2: Service quality and reliability penalty (uncovered demand + maintenance + timing)
    f2 = (bd["uncovered_demand_passengers"] * 2.0
          + bd["maintenance_violation_km"] * 5.0
          + bd["timing_violations"] * 1000.0)
    return (f1, f2), cost, bd

# Domination check: returns True if objective vector p dominates vector q
def dominates(p, q):
    return (p[0] <= q[0] and p[1] <= q[1]) and (p[0] < q[0] or p[1] < q[1])

# Fast non-dominated sorting
def fast_non_dominated_sort(objectives):
    num_ind = len(objectives)
    domination_count = [0] * num_ind
    dominated_solutions = [[] for _ in range(num_ind)]
    fronts = [[]]

    for p in range(num_ind):
        for q in range(num_ind):
            if dominates(objectives[p], objectives[q]):
                dominated_solutions[p].append(q)
            elif dominates(objectives[q], objectives[p]):
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

# Calculate crowding distance for diversity maintenance
def calculate_crowding_distance(front, objectives):
    distance = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        for idx in front:
            distance[idx] = float("inf")
        return distance

    for m in range(2):
        front_sorted = sorted(front, key=lambda idx: objectives[idx][m])
        distance[front_sorted[0]] = float("inf")
        distance[front_sorted[-1]] = float("inf")
        norm = objectives[front_sorted[-1]][m] - objectives[front_sorted[0]][m]
        if norm == 0:
            continue
        for i in range(1, len(front_sorted) - 1):
            distance[front_sorted[i]] += (objectives[front_sorted[i + 1]][m] - objectives[front_sorted[i - 1]][m]) / norm
    return distance

# Mutation operator
def mutate(chains, all_unit_ids, all_trip_ids, rng):
    mutated = copy_chains(chains)
    if rng.random() < 0.5:
        donors = [uid for uid, c in mutated.items() if c]
        if donors:
            d = rng.choice(donors)
            t = rng.choice(mutated[d])
            r = rng.choice(all_unit_ids)
            mutated[d].remove(t)
            mutated[r].append(t)
    else:
        non_empty = [uid for uid, c in mutated.items() if c]
        if len(non_empty) >= 2:
            u1, u2 = rng.sample(non_empty, 2)
            t1 = rng.choice(mutated[u1])
            t2 = rng.choice(mutated[u2])
            mutated[u1].remove(t1)
            mutated[u2].remove(t2)
            mutated[u1].append(t2)
            mutated[u2].append(t1)
    return mutated

# Uniform crossover between two schedules
def crossover(p1, p2, all_trip_ids, all_unit_ids, rng):
    child = {uid: [] for uid in all_unit_ids}
    t_map1 = {t: [uid for uid, c in p1.items() if t in c] for t in all_trip_ids}
    t_map2 = {t: [uid for uid, c in p2.items() if t in c] for t in all_trip_ids}
    for t in all_trip_ids:
        chosen = t_map1[t] if rng.random() < 0.5 else t_map2[t]
        if not chosen:
            chosen = [rng.choice(all_unit_ids)]
        for uid in chosen:
            child[uid].append(t)
    return child

# Crowded tournament comparison
def crowded_comparison(i1, i2, rank, crowding):
    if rank[i1] < rank[i2]:
        return i1
    if rank[i2] < rank[i1]:
        return i2
    if crowding[i1] > crowding[i2]:
        return i1
    return i2

# Core NSGA-II solver
def solve_nsga2(fleet, trips, dist_lookup, maint_lookup,
                pop_size=POPULATION_SIZE, generations=GENERATIONS, seed=SEED):
    rng = random.Random(seed)
    fleet_lookup, trips_lookup = get_lookups(fleet, trips)
    all_unit_ids = list(fleet_lookup.keys())
    all_trip_ids = list(trips_lookup.keys())

    # Build seed individual using greedy construction
    seed_chains, _ = greedy_construction(fleet, trips, dist_lookup)
    population = [seed_chains]

    # Initialize population by perturbing greedy schedule
    for _ in range(pop_size - 1):
        ind = copy_chains(seed_chains)
        for _ in range(rng.randint(2, 5)):
            ind = mutate(ind, all_unit_ids, all_trip_ids, rng)
        population.append(ind)

    history = []

    # Evolution loop
    for gen in range(generations):
        # Evaluate current population
        eval_results = [
            evaluate_objectives(ind, fleet_lookup, trips_lookup, dist_lookup, maint_lookup)
            for ind in population
        ]
        objectives = [r[0] for r in eval_results]
        costs = [r[1] for r in eval_results]
        breakdowns = [r[2] for r in eval_results]

        # Fast non-dominated sorting and crowding distance assignment
        fronts = fast_non_dominated_sort(objectives)
        rank = {}
        crowding = {}
        for r_idx, front in enumerate(fronts):
            cd = calculate_crowding_distance(front, objectives)
            for idx in front:
                rank[idx] = r_idx
                crowding[idx] = cd[idx]

        # Best compromise solution by minimal scalarized total cost
        min_cost_idx = min(range(len(population)), key=lambda i: costs[i])
        history.append({
            "step": gen,
            "best_cost": round(costs[min_cost_idx], 1),
            "f1_op_cost": round(objectives[min_cost_idx][0], 1),
            "f2_penalty": round(objectives[min_cost_idx][1], 1)
        })

        # Offspring generation
        offspring = []
        while len(offspring) < pop_size:
            p1_idx = crowded_comparison(rng.randint(0, pop_size - 1), rng.randint(0, pop_size - 1), rank, crowding)
            p2_idx = crowded_comparison(rng.randint(0, pop_size - 1), rng.randint(0, pop_size - 1), rank, crowding)
            child = crossover(population[p1_idx], population[p2_idx], all_trip_ids, all_unit_ids, rng)
            if rng.random() < 0.4:
                child = mutate(child, all_unit_ids, all_trip_ids, rng)
            offspring.append(child)

        # Merge parents and offspring (2N population)
        combined_pop = population + offspring
        combined_eval = [
            evaluate_objectives(ind, fleet_lookup, trips_lookup, dist_lookup, maint_lookup)
            for ind in combined_pop
        ]
        combined_objs = [r[0] for r in combined_eval]
        combined_fronts = fast_non_dominated_sort(combined_objs)

        # Select next generation through elitist front truncation
        next_pop = []
        for front in combined_fronts:
            if len(next_pop) + len(front) <= pop_size:
                next_pop.extend([combined_pop[i] for i in front])
            else:
                cd = calculate_crowding_distance(front, combined_objs)
                sorted_front = sorted(front, key=lambda idx: cd[idx], reverse=True)
                needed = pop_size - len(next_pop)
                next_pop.extend([combined_pop[i] for i in sorted_front[:needed]])
                break
        population = next_pop

    # Final evaluation of population
    final_eval = [
        evaluate_objectives(ind, fleet_lookup, trips_lookup, dist_lookup, maint_lookup)
        for ind in population
    ]
    final_objs = [r[0] for r in final_eval]
    final_costs = [r[1] for r in final_eval]
    final_breakdowns = [r[2] for r in final_eval]
    final_fronts = fast_non_dominated_sort(final_objs)

    # First Pareto front solutions
    pareto_indices = final_fronts[0]
    pareto_points = [
        {"solution_index": idx, "f1_op_cost": final_objs[idx][0], "f2_penalty": final_objs[idx][1], "total_cost": final_costs[idx]}
        for idx in pareto_indices
    ]

    # Best compromise selection
    best_idx = min(range(len(population)), key=lambda i: final_costs[i])
    best_chains = copy_chains(population[best_idx])
    best_cost = final_costs[best_idx]
    best_breakdown = final_breakdowns[best_idx]

    return best_chains, best_cost, best_breakdown, pareto_points, history

# Standalone execution entrypoint
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    stations, dist_lookup, fleet, trips, maint_lookup, meta = load_instance()

    # Run NSGA-II solver
    best_chains, best_cost, best_breakdown, pareto_points, history = solve_nsga2(
        fleet, trips, dist_lookup, maint_lookup
    )

    print("NSGA-II best compromise solution cost breakdown:")
    print(json.dumps(best_breakdown, indent=2))

    # Save schedule dataframe
    schedule_df = chains_to_schedule_df(best_chains, trips)
    schedule_path = os.path.join(OUT_DIR, "final_schedule.csv")
    schedule_df.to_csv(schedule_path, index=False)

    # Save Pareto front records
    pareto_df = pd.DataFrame(pareto_points)
    pareto_path = os.path.join(OUT_DIR, "pareto_front.csv")
    pareto_df.to_csv(pareto_path, index=False)

    # Save convergence history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(OUT_DIR, "nsga2_cost_history.csv")
    history_df.to_csv(history_path, index=False)

    # Save convergence plot
    plot_path = os.path.join(OUT_DIR, "nsga2_convergence.png")
    plot_cost_history(history, save_path=plot_path, title="NSGA-II Convergence")

    # Plot Pareto front scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pareto_df["f1_op_cost"], pareto_df["f2_penalty"], color="#d62728", s=40, edgecolors="k", zorder=3)
    ax.set_title("NSGA-II Pareto Optimal Front", fontsize=12, fontweight="bold")
    ax.set_xlabel("Operational Resource Cost (f1)", fontsize=10)
    ax.set_ylabel("Penalty & Reliability Cost (f2)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    pareto_img_path = os.path.join(OUT_DIR, "pareto_front.png")
    plt.savefig(pareto_img_path, dpi=300)
    plt.close(fig)

    # Save summary json
    summary_path = os.path.join(OUT_DIR, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"breakdown": best_breakdown, "pareto_count": len(pareto_points), "algorithm": "nsga2"}, f, indent=2)

    print(f"Saved: {schedule_path}")
    print(f"Saved: {pareto_path}")
    print(f"Saved: {summary_path}")
