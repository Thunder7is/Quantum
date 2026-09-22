# Simulated annealing metaheuristic solver for rolling stock scheduling
import os
import math
import json
import random
import pandas as pd
from ..utils.data_loader import load_instance, get_lookups
from ..utils.cost import compute_cost
from ..utils.greedy_init import greedy_construction, chains_to_schedule_df
from ..utils.viz import plot_cost_history

# Default SA hyperparameters
INITIAL_TEMP = 500.0
COOLING_RATE = 0.995
MIN_TEMP = 0.5
ITERATIONS_PER_TEMP = 40
SEED = 7

# Directory for simulated annealing output artifacts
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "sa"))

# Deep copy helper for chain dictionary
def copy_chains(chains):
    return {uid: list(trips) for uid, trips in chains.items()}

# Relocate move: transfer a trip from one unit to another
def move_relocate(chains, all_unit_ids, rng):
    new_chains = copy_chains(chains)
    donors = [uid for uid, c in new_chains.items() if c]
    if not donors:
        return new_chains
    donor = rng.choice(donors)
    trip_id = rng.choice(new_chains[donor])
    receiver = rng.choice(all_unit_ids)
    new_chains[donor].remove(trip_id)
    new_chains[receiver].append(trip_id)
    return new_chains

# Swap move: exchange trips between two distinct units
def move_swap(chains, rng):
    new_chains = copy_chains(chains)
    non_empty = [uid for uid, c in new_chains.items() if c]
    if len(non_empty) < 2:
        return new_chains
    u1, u2 = rng.sample(non_empty, 2)
    t1 = rng.choice(new_chains[u1])
    t2 = rng.choice(new_chains[u2])
    new_chains[u1].remove(t1)
    new_chains[u2].remove(t2)
    new_chains[u1].append(t2)
    new_chains[u2].append(t1)
    return new_chains

# Couple/Decouple move: adjust helper unit assignment for a trip
def move_couple_decouple(chains, all_unit_ids, all_trip_ids, rng):
    new_chains = copy_chains(chains)
    trip_id = rng.choice(all_trip_ids)
    assigned_units = [uid for uid, c in new_chains.items() if trip_id in c]
    if rng.random() < 0.5 or not assigned_units:
        # Add helper unit from available candidate pool
        candidates = [u for u in all_unit_ids if u not in assigned_units]
        if candidates:
            uid = rng.choice(candidates)
            new_chains[uid].append(trip_id)
    else:
        # Remove helper unit if multiple units are currently attached
        if len(assigned_units) > 1:
            uid = rng.choice(assigned_units)
            new_chains[uid].remove(trip_id)
    return new_chains

# Core simulated annealing optimization loop
def solve_simulated_annealing(fleet, trips, dist_lookup, maint_lookup, init_chains=None,
                              initial_temp=INITIAL_TEMP, cooling_rate=COOLING_RATE,
                              min_temp=MIN_TEMP, iterations_per_temp=ITERATIONS_PER_TEMP, seed=SEED):
    rng = random.Random(seed)
    fleet_lookup, trips_lookup = get_lookups(fleet, trips)
    all_unit_ids = list(fleet_lookup.keys())
    all_trip_ids = list(trips_lookup.keys())

    # Build initial greedy solution if none provided
    if init_chains is None:
        init_chains, _ = greedy_construction(fleet, trips, dist_lookup)

    current = copy_chains(init_chains)
    current_cost, current_breakdown = compute_cost(current, fleet_lookup, trips_lookup, dist_lookup, maint_lookup)

    best = copy_chains(current)
    best_cost = current_cost
    best_breakdown = current_breakdown

    temp = initial_temp
    history = []
    step = 0

    # Execute annealing schedule until threshold reached
    while temp > min_temp:
        for _ in range(iterations_per_temp):
            # Select random neighborhood operator
            operator_choice = rng.random()
            if operator_choice < 0.4:
                candidate = move_relocate(current, all_unit_ids, rng)
            elif operator_choice < 0.8:
                candidate = move_swap(current, rng)
            else:
                candidate = move_couple_decouple(current, all_unit_ids, all_trip_ids, rng)

            cand_cost, cand_breakdown = compute_cost(candidate, fleet_lookup, trips_lookup, dist_lookup, maint_lookup)
            delta = cand_cost - current_cost

            # Metropolis acceptance criterion
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                current = candidate
                current_cost = cand_cost
                current_breakdown = cand_breakdown
                if current_cost < best_cost:
                    best = copy_chains(current)
                    best_cost = current_cost
                    best_breakdown = current_breakdown

            step += 1

        history.append({
            "step": step,
            "temp": round(temp, 2),
            "current_cost": round(current_cost, 1),
            "best_cost": round(best_cost, 1)
        })
        temp *= cooling_rate

    return best, best_cost, best_breakdown, history

# Standalone execution entrypoint
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    stations, dist_lookup, fleet, trips, maint_lookup, meta = load_instance()
    fleet_lookup, trips_lookup = get_lookups(fleet, trips)

    # Initial greedy baseline evaluation
    init_chains, _ = greedy_construction(fleet, trips, dist_lookup)
    init_cost, init_breakdown = compute_cost(init_chains, fleet_lookup, trips_lookup, dist_lookup, maint_lookup)
    print("Initial (greedy) solution cost breakdown:")
    print(json.dumps(init_breakdown, indent=2))

    # Run simulated annealing solver
    best_chains, best_cost, best_breakdown, history = solve_simulated_annealing(
        fleet, trips, dist_lookup, maint_lookup, init_chains=init_chains
    )

    print("\nFinal (SA-optimized) solution cost breakdown:")
    print(json.dumps(best_breakdown, indent=2))
    improvement = 100 * (init_cost - best_cost) / init_cost if init_cost else 0
    print(f"\nImprovement over greedy baseline: {improvement:.1f}%")

    # Save output artifacts
    schedule_df = chains_to_schedule_df(best_chains, trips)
    schedule_path = os.path.join(OUT_DIR, "final_schedule.csv")
    schedule_df.to_csv(schedule_path, index=False)

    history_df = pd.DataFrame(history)
    history_path = os.path.join(OUT_DIR, "sa_cost_history.csv")
    history_df.to_csv(history_path, index=False)

    plot_path = os.path.join(OUT_DIR, "sa_convergence.png")
    plot_cost_history(history, save_path=plot_path, title="Simulated Annealing Convergence")

    summary_path = os.path.join(OUT_DIR, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "initial_breakdown": init_breakdown,
            "final_breakdown": best_breakdown,
            "improvement_pct": round(improvement, 1),
            "algorithm": "simulated_annealing"
        }, f, indent=2)

    print(f"Saved: {schedule_path}")
    print(f"Saved: {history_path}")
    print(f"Saved: {summary_path}")
