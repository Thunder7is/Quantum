# Genetic algorithm solver for rolling stock scheduling optimization
import os
import json
import random
import pandas as pd
from ..utils.data_loader import load_instance, get_lookups
from ..utils.cost import compute_cost
from ..utils.greedy_init import greedy_construction, chains_to_schedule_df
from ..utils.viz import plot_cost_history

# Default GA hyperparameters
POPULATION_SIZE = 30
GENERATIONS = 60
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.35
CROSSOVER_RATE = 0.8
SEED = 42

# Output directory path for genetic algorithm results
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "ga"))

# Clone solution chains structure
def copy_chains(chains):
    return {uid: list(trips) for uid, trips in chains.items()}

# Mutate schedule via perturbation operators
def mutate(chains, all_unit_ids, all_trip_ids, rng):
    mutated = copy_chains(chains)
    op = rng.random()
    if op < 0.45:
        # Relocate random trip
        donors = [uid for uid, c in mutated.items() if c]
        if donors:
            d = rng.choice(donors)
            t = rng.choice(mutated[d])
            r = rng.choice(all_unit_ids)
            mutated[d].remove(t)
            mutated[r].append(t)
    elif op < 0.8:
        # Swap trips between two units
        non_empty = [uid for uid, c in mutated.items() if c]
        if len(non_empty) >= 2:
            u1, u2 = rng.sample(non_empty, 2)
            t1 = rng.choice(mutated[u1])
            t2 = rng.choice(mutated[u2])
            mutated[u1].remove(t1)
            mutated[u2].remove(t2)
            mutated[u1].append(t2)
            mutated[u2].append(t1)
    else:
        # Couple or decouple helper unit on random trip
        t = rng.choice(all_trip_ids)
        assigned = [uid for uid, c in mutated.items() if t in c]
        if rng.random() < 0.5 or not assigned:
            cands = [u for u in all_unit_ids if u not in assigned]
            if cands:
                mutated[rng.choice(cands)].append(t)
        elif len(assigned) > 1:
            mutated[rng.choice(assigned)].remove(t)
    return mutated

# Recombine assignments from two parent schedules
def crossover(p1, p2, all_trip_ids, all_unit_ids, rng):
    child = {uid: [] for uid in all_unit_ids}
    # Invert parent representations to trip-to-units mappings
    t_map1 = {t: [uid for uid, c in p1.items() if t in c] for t in all_trip_ids}
    t_map2 = {t: [uid for uid, c in p2.items() if t in c] for t in all_trip_ids}

    # For each trip, inherit assignment from parent 1 or 2
    for t in all_trip_ids:
        chosen_units = t_map1[t] if rng.random() < 0.5 else t_map2[t]
        # Guarantee at least one unit assigned per trip
        if not chosen_units:
            chosen_units = [rng.choice(all_unit_ids)]
        for uid in chosen_units:
            child[uid].append(t)
    return child

# Tournament selection for picking parents
def tournament_selection(population_with_costs, tournament_size, rng):
    sampled = rng.sample(population_with_costs, tournament_size)
    # Pick candidate with lowest cost
    sampled.sort(key=lambda item: item[1])
    return sampled[0][0]

# Core genetic algorithm optimization solver
def solve_genetic_algorithm(fleet, trips, dist_lookup, maint_lookup,
                             pop_size=POPULATION_SIZE, generations=GENERATIONS,
                             seed=SEED):
    rng = random.Random(seed)
    fleet_lookup, trips_lookup = get_lookups(fleet, trips)
    all_unit_ids = list(fleet_lookup.keys())
    all_trip_ids = list(trips_lookup.keys())

    # Build seed individual using greedy construction
    seed_chains, _ = greedy_construction(fleet, trips, dist_lookup)
    population = [seed_chains]

    # Generate initial diverse population by repeated perturbation
    for _ in range(pop_size - 1):
        ind = copy_chains(seed_chains)
        num_mutations = rng.randint(2, 6)
        for _ in range(num_mutations):
            ind = mutate(ind, all_unit_ids, all_trip_ids, rng)
        population.append(ind)

    # Evaluate initial population
    pop_evaluated = []
    for ind in population:
        c, b = compute_cost(ind, fleet_lookup, trips_lookup, dist_lookup, maint_lookup)
        pop_evaluated.append((ind, c, b))

    pop_evaluated.sort(key=lambda x: x[1])
    best_chains = copy_chains(pop_evaluated[0][0])
    best_cost = pop_evaluated[0][1]
    best_breakdown = pop_evaluated[0][2]

    history = [{"step": 0, "best_cost": round(best_cost, 1), "current_cost": round(pop_evaluated[0][1], 1)}]

    # Evolve population across generations
    for gen in range(1, generations + 1):
        new_pop = [copy_chains(best_chains)]  # Elitism preservation

        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop_evaluated, TOURNAMENT_SIZE, rng)
            p2 = tournament_selection(pop_evaluated, TOURNAMENT_SIZE, rng)

            if rng.random() < CROSSOVER_RATE:
                child = crossover(p1, p2, all_trip_ids, all_unit_ids, rng)
            else:
                child = copy_chains(p1)

            if rng.random() < MUTATION_RATE:
                child = mutate(child, all_unit_ids, all_trip_ids, rng)

            new_pop.append(child)

        # Evaluate offspring generation
        pop_evaluated = []
        for ind in new_pop:
            c, b = compute_cost(ind, fleet_lookup, trips_lookup, dist_lookup, maint_lookup)
            pop_evaluated.append((ind, c, b))

        pop_evaluated.sort(key=lambda x: x[1])
        if pop_evaluated[0][1] < best_cost:
            best_chains = copy_chains(pop_evaluated[0][0])
            best_cost = pop_evaluated[0][1]
            best_breakdown = pop_evaluated[0][2]

        history.append({
            "step": gen,
            "best_cost": round(best_cost, 1),
            "current_cost": round(pop_evaluated[0][1], 1)
        })

    return best_chains, best_cost, best_breakdown, history

# Standalone execution entrypoint
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    stations, dist_lookup, fleet, trips, maint_lookup, meta = load_instance()

    # Run genetic algorithm solver
    best_chains, best_cost, best_breakdown, history = solve_genetic_algorithm(
        fleet, trips, dist_lookup, maint_lookup
    )

    print("Genetic Algorithm solution cost breakdown:")
    print(json.dumps(best_breakdown, indent=2))

    # Save schedule dataframe
    schedule_df = chains_to_schedule_df(best_chains, trips)
    schedule_path = os.path.join(OUT_DIR, "final_schedule.csv")
    schedule_df.to_csv(schedule_path, index=False)

    # Save convergence history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(OUT_DIR, "ga_cost_history.csv")
    history_df.to_csv(history_path, index=False)

    # Save convergence plot
    plot_path = os.path.join(OUT_DIR, "ga_convergence.png")
    plot_cost_history(history, save_path=plot_path, title="Genetic Algorithm Convergence")

    # Save summary json
    summary_path = os.path.join(OUT_DIR, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"breakdown": best_breakdown, "algorithm": "genetic_algorithm"}, f, indent=2)

    print(f"Saved: {schedule_path}")
    print(f"Saved: {history_path}")
    print(f"Saved: {summary_path}")
