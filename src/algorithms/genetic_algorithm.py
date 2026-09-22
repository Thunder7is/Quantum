# Genetic algorithm solver for rolling stock scheduling optimization
import os
import sys
import json
import random
import pandas as pd

# Support execution as direct script or package module
try:
    from ..utils.data_loader import load_instance, get_lookups, travel
    from ..utils.cost import compute_cost
    from ..utils.greedy_init import greedy_construction, chains_to_schedule_df
    from ..utils.viz import plot_cost_history
except (ImportError, ValueError):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.utils.data_loader import load_instance, get_lookups, travel
    from src.utils.cost import compute_cost
    from src.utils.greedy_init import greedy_construction, chains_to_schedule_df
    from src.utils.viz import plot_cost_history

# GA hyperparameters matching research specification
POPULATION_SIZE = 80
MAX_GENERATIONS = 300
PLATEAU_PATIENCE = 50
TOURNAMENT_K = 5
CROSSOVER_PROB = 0.85
MUTATION_PROB_GENE = 0.1
ELITISM_COUNT = 2
TIMING_PENALTY_MULTIPLIER = 10.0
SEED = 42
MAX_COUPLE = 2

# Directory path for genetic algorithm outputs
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "ga"))

# Map discrete trip-to-unit chromosome representation into unit circulation chains
def chromosome_to_chains(chromosome: list, fleet_list: list, trips_list: list, type_to_unit_indices: dict) -> dict:
    chains = {u["unit_id"]: [] for u in fleet_list}
    for trip_idx, u_idx in enumerate(chromosome):
        trip = trips_list[trip_idx]
        primary_unit = fleet_list[u_idx]
        assigned_uids = [primary_unit["unit_id"]]

        # Add second unit from same rolling stock type pool if demand exceeds single unit capacity
        if trip.get("expected_passenger_demand", 0) > primary_unit["capacity"]:
            candidates = [idx for idx in type_to_unit_indices[primary_unit["unit_type"]] if idx != u_idx]
            if candidates:
                # Deterministic selection based on trip and unit index
                sec_idx = candidates[(trip_idx + u_idx) % len(candidates)]
                sec_uid = fleet_list[sec_idx]["unit_id"]
                # Enforce max coupling limit per trip
                if len(assigned_uids) < MAX_COUPLE and sec_uid not in assigned_uids:
                    assigned_uids.append(sec_uid)

        for uid in assigned_uids:
            chains[uid].append(trip["trip_id"])

    return chains

# Evaluate chromosome fitness using compute_cost with 10x timing violation penalty
def evaluate_chromosome(chromosome: list, fleet_list: list, trips_list: list, type_to_unit_indices: dict,
                        trips_lookup: dict, fleet_lookup: dict, dist_lookup: dict, maint_lookup: dict):
    chains = chromosome_to_chains(chromosome, fleet_list, trips_list, type_to_unit_indices)
    breakdown = compute_cost(chains, trips_lookup, fleet_lookup, dist_lookup, maint_lookup)
    raw_cost = breakdown["total"]
    timing_violations = breakdown["timing_violations"]

    # Apply 10x penalty multiplier to timing violation term
    timing_extra = 9.0 * 1000.0 * timing_violations
    fitness = raw_cost + timing_extra

    return fitness, raw_cost, breakdown, chains

# Perform tournament selection with tournament size k
def tournament_selection(population_eval: list, k: int, rng: random.Random) -> list:
    sample = rng.sample(population_eval, k)
    # Lower fitness score is better
    sample.sort(key=lambda item: item[0])
    return sample[0][1]

# Partially Mapped Crossover with cut points and coupling repair
def pmx_crossover(parent1: list, parent2: list, num_units: int, rng: random.Random):
    length = len(parent1)
    if length < 2 or rng.random() > CROSSOVER_PROB:
        return list(parent1), list(parent2)

    # Select two cut points
    c1 = rng.randint(0, length - 2)
    c2 = rng.randint(c1 + 1, length - 1)

    child1 = list(parent1)
    child2 = list(parent2)

    # Swap interior segment
    child1[c1:c2 + 1] = parent2[c1:c2 + 1]
    child2[c1:c2 + 1] = parent1[c1:c2 + 1]

    # Repair valid range for unit indices
    for i in range(length):
        child1[i] = max(0, min(num_units - 1, child1[i]))
        child2[i] = max(0, min(num_units - 1, child2[i]))

    return child1, child2

# Per-gene uniform random mutation
def mutate_chromosome(chromosome: list, num_units: int, mutation_prob: float, rng: random.Random) -> list:
    mutated = list(chromosome)
    for i in range(len(mutated)):
        if rng.random() < mutation_prob:
            mutated[i] = rng.randint(0, num_units - 1)
    return mutated

# Core genetic algorithm optimization solver
def solve_genetic_algorithm(fleet, trips, dist_lookup, maint_lookup,
                             pop_size=POPULATION_SIZE, max_generations=MAX_GENERATIONS,
                             plateau_patience=PLATEAU_PATIENCE, seed=SEED):
    rng = random.Random(seed)
    fleet_lookup, trips_lookup = get_lookups(fleet, trips)
    trips_sorted = trips.sort_values("departure_min").reset_index(drop=True)
    trips_list = trips_sorted.to_dict("records")
    fleet_list = fleet.to_dict("records")
    num_units = len(fleet_list)
    num_trips = len(trips_list)

    # Group unit indices by rolling stock type for consistent coupling
    type_to_unit_indices = {}
    for idx, u in enumerate(fleet_list):
        type_to_unit_indices.setdefault(u["unit_type"], []).append(idx)

    # Initialize population by randomly permuting unit assignments
    population = []
    for _ in range(pop_size):
        chromosome = [rng.randint(0, num_units - 1) for _ in range(num_trips)]
        population.append(chromosome)

    # Seed one chromosome using greedy construction heuristic if available
    inst = {"fleet": fleet, "trips": trips, "dist_lookup": dist_lookup, "maint_lookup": maint_lookup}
    greedy_chains = greedy_construction(inst)
    greedy_chrom = [0] * num_trips
    uid_to_idx = {u["unit_id"]: idx for idx, u in enumerate(fleet_list)}
    for uid, c in greedy_chains.items():
        for tid in c:
            for i, t in enumerate(trips_list):
                if t["trip_id"] == tid:
                    greedy_chrom[i] = uid_to_idx[uid]
    population[0] = greedy_chrom

    # Evaluate initial population
    pop_eval = []
    for chrom in population:
        fit, raw_cost, bd, chains = evaluate_chromosome(
            chrom, fleet_list, trips_list, type_to_unit_indices,
            trips_lookup, fleet_lookup, dist_lookup, maint_lookup
        )
        pop_eval.append((fit, chrom, raw_cost, bd, chains))

    pop_eval.sort(key=lambda item: item[0])
    best_overall = pop_eval[0]
    best_fitness = best_overall[0]
    patience_counter = 0

    cost_history = []
    generations_with_timing_violations = []
    trip_feasible_tracker = {t["trip_id"]: False for t in trips_list}

    # Record generation 0 metrics
    avg_cost_gen0 = sum(item[2] for item in pop_eval) / len(pop_eval)
    cost_history.append({
        "generation": 0,
        "step": 0,
        "best_cost": round(best_overall[2], 1),
        "avg_cost": round(avg_cost_gen0, 1),
        "total_cost": round(best_overall[2], 1)
    })
    if best_overall[3]["timing_violations"] > 0:
        generations_with_timing_violations.append(0)

    # Evolution loop
    for gen in range(1, max_generations + 1):
        # Elitism: retain top ELITISM_COUNT chromosomes
        next_population = [item[1] for item in pop_eval[:ELITISM_COUNT]]

        # Generate offspring via selection, crossover, and mutation
        while len(next_population) < pop_size:
            p1 = tournament_selection(pop_eval, TOURNAMENT_K, rng)
            p2 = tournament_selection(pop_eval, TOURNAMENT_K, rng)

            c1, c2 = pmx_crossover(p1, p2, num_units, rng)
            c1 = mutate_chromosome(c1, num_units, MUTATION_PROB_GENE, rng)
            c2 = mutate_chromosome(c2, num_units, MUTATION_PROB_GENE, rng)

            next_population.append(c1)
            if len(next_population) < pop_size:
                next_population.append(c2)

        population = next_population

        # Evaluate offspring generation
        pop_eval = []
        for chrom in population:
            fit, raw_cost, bd, chains = evaluate_chromosome(
                chrom, fleet_list, trips_list, type_to_unit_indices,
                trips_lookup, fleet_lookup, dist_lookup, maint_lookup
            )
            pop_eval.append((fit, chrom, raw_cost, bd, chains))

        pop_eval.sort(key=lambda item: item[0])
        gen_best = pop_eval[0]
        gen_avg_cost = sum(item[2] for item in pop_eval) / len(pop_eval)

        # Track timing violations in current generation
        if gen_best[3]["timing_violations"] > 0:
            generations_with_timing_violations.append(gen)

        # Track feasible assignment coverage per trip
        for uid, chain in gen_best[4].items():
            for tid in chain:
                if gen_best[3]["timing_violations"] == 0:
                    trip_feasible_tracker[tid] = True

        # Check fitness improvement for convergence termination
        if gen_best[0] < best_fitness - 1e-4:
            best_fitness = gen_best[0]
            best_overall = gen_best
            patience_counter = 0
        else:
            patience_counter += 1

        cost_history.append({
            "generation": gen,
            "step": gen,
            "best_cost": round(best_overall[2], 1),
            "avg_cost": round(gen_avg_cost, 1),
            "total_cost": round(best_overall[2], 1)
        })

        # Plateau termination check
        if patience_counter >= plateau_patience:
            break

    # Extract non-feasibly assigned trips
    never_feasibly_assigned = [tid for tid, feasible in trip_feasible_tracker.items() if not feasible]

    best_chains = best_overall[4]
    best_cost = best_overall[2]
    best_breakdown = best_overall[3]

    diagnostics = {
        "generations_with_timing_violations": generations_with_timing_violations,
        "trips_never_feasibly_assigned": never_feasibly_assigned,
        "completed_generations": len(cost_history) - 1,
    }

    return best_chains, best_cost, best_breakdown, cost_history, diagnostics

# Main execution routine
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    inst = load_instance()
    fleet, trips, dist_lookup, maint_lookup = inst["fleet"], inst["trips"], inst["dist_lookup"], inst["maint_lookup"]

    print("=================================================================")
    print("  Genetic Algorithm Solver (Evolutionary Optimization)")
    print(f"  Fleet: {len(fleet)} units | Trips: {len(trips)} | Population: {POPULATION_SIZE}")
    print("=================================================================\n")

    # Run genetic algorithm solver
    best_chains, best_cost, best_breakdown, cost_history, diagnostics = solve_genetic_algorithm(
        fleet, trips, dist_lookup, maint_lookup
    )

    print("Genetic Algorithm solution cost breakdown:")
    print(json.dumps(best_breakdown, indent=2))
    print(f"\nCompleted generations: {diagnostics['completed_generations']}")
    print(f"Generations with timing violations: {len(diagnostics['generations_with_timing_violations'])}")
    print(f"Trips never feasibly assigned: {len(diagnostics['trips_never_feasibly_assigned'])}")

    # Save final schedule CSV
    schedule_df = chains_to_schedule_df(best_chains, trips)
    schedule_path = os.path.join(OUT_DIR, "final_schedule.csv")
    schedule_df.to_csv(schedule_path, index=False)

    # Save GA cost history CSV with generation, best_cost, avg_cost
    history_df = pd.DataFrame(cost_history)[["generation", "best_cost", "avg_cost"]]
    history_path = os.path.join(OUT_DIR, "ga_cost_history.csv")
    history_df.to_csv(history_path, index=False)

    # Save cost history PNG via plot_cost_history
    plot_path = os.path.join(OUT_DIR, "cost_history.png")
    plot_cost_history(cost_history, "Genetic Algorithm", plot_path)

    # Save run summary JSON
    summary_path = os.path.join(OUT_DIR, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "breakdown": best_breakdown,
            "generations": diagnostics["completed_generations"],
            "hyperparameters": {
                "population_size": POPULATION_SIZE,
                "max_generations": MAX_GENERATIONS,
                "plateau_patience": PLATEAU_PATIENCE,
                "tournament_k": TOURNAMENT_K,
                "crossover_prob": CROSSOVER_PROB,
                "mutation_prob": MUTATION_PROB_GENE,
                "elitism": ELITISM_COUNT,
            },
            "algorithm": "genetic_algorithm"
        }, f, indent=2)

    # Save failure analysis JSON
    failure_path = os.path.join(OUT_DIR, "failure_analysis.json")
    with open(failure_path, "w") as f:
        json.dump({
            "generations_with_timing_violations": diagnostics["generations_with_timing_violations"],
            "trips_never_feasibly_assigned": diagnostics["trips_never_feasibly_assigned"],
        }, f, indent=2)

    print(f"\nSaved schedule: {schedule_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved failure analysis: {failure_path}")

# Standalone execution entrypoint
if __name__ == "__main__":
    main()
