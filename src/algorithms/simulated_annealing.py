# Simulated annealing metaheuristic solver for rolling stock scheduling
import os
import sys
import math
import json
import random
import pandas as pd

# Support execution as direct script or package module
try:
    from ..utils.data_loader import load_instance, get_lookups
    from ..utils.cost import compute_cost
    from ..utils.greedy_init import greedy_construction, chains_to_schedule_df
    from ..utils.feasibility import is_feasible_arc
    from ..utils.viz import plot_cost_history
except (ImportError, ValueError):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.utils.data_loader import load_instance, get_lookups
    from src.utils.cost import compute_cost
    from src.utils.greedy_init import greedy_construction, chains_to_schedule_df
    from src.utils.feasibility import is_feasible_arc
    from src.utils.viz import plot_cost_history

# Default SA hyperparameters specified for research benchmark
INITIAL_TEMP = 500.0
COOLING_RATE = 0.995
MIN_TEMP = 0.5
ITERATIONS_PER_TEMP = 40
SEED = 7

# Directory path for simulated annealing outputs
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "sa"))

# Deep copy helper for chain dictionary
def copy_chains(chains: dict) -> dict:
    return {uid: list(trips) for uid, trips in chains.items()}

# Relocate move: transfer a trip to a candidate unit with feasible arc verification
def move_relocate(chains, all_unit_ids, trips_lookup, dist_lookup, fleet_lookup, rng):
    new_chains = copy_chains(chains)
    donors = [uid for uid, c in new_chains.items() if c]
    if not donors:
        return new_chains
    donor = rng.choice(donors)
    trip_id = rng.choice(new_chains[donor])
    trip = trips_lookup[trip_id]

    # Sample candidate receiver units and prioritize those with feasible transition arcs
    receivers = list(all_unit_ids)
    rng.shuffle(receivers)
    selected_receiver = receivers[0]

    for r_uid in receivers:
        r_chain = new_chains[r_uid]
        if not r_chain:
            selected_receiver = r_uid
            break
        # Sort existing trips on receiver unit to check adjacency
        sorted_r = sorted(r_chain, key=lambda tid: trips_lookup[tid]["departure_min"])
        unit_row = fleet_lookup[r_uid]
        # Check arc feasibility against adjacent trips in receiver chain
        feasible = True
        for existing_tid in sorted_r:
            existing_trip = trips_lookup[existing_tid]
            if existing_trip["departure_min"] < trip["departure_min"]:
                if not is_feasible_arc(unit_row, existing_trip, trip, dist_lookup):
                    feasible = False
                    break
            elif trip["departure_min"] < existing_trip["departure_min"]:
                if not is_feasible_arc(unit_row, trip, existing_trip, dist_lookup):
                    feasible = False
                    break
        if feasible:
            selected_receiver = r_uid
            break

    new_chains[donor].remove(trip_id)
    new_chains[selected_receiver].append(trip_id)
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

# Couple/Decouple move: adjust consist helper unit assignment for a trip
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
        # Remove helper unit if multiple units are currently coupled
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
        init_instance = {"fleet": fleet, "trips": trips, "dist_lookup": dist_lookup, "maint_lookup": maint_lookup}
        init_chains = greedy_construction(init_instance)

    current = copy_chains(init_chains)
    current_breakdown = compute_cost(current, trips_lookup, fleet_lookup, dist_lookup, maint_lookup)
    current_cost = current_breakdown["total"]

    best = copy_chains(current)
    best_cost = current_cost
    best_breakdown = current_breakdown

    temp = initial_temp
    history = []
    step = 0

    # Execute annealing schedule until threshold reached
    while temp > min_temp:
        for _ in range(iterations_per_temp):
            # Select random neighborhood perturbation operator
            operator_choice = rng.random()
            if operator_choice < 0.4:
                candidate = move_relocate(current, all_unit_ids, trips_lookup, dist_lookup, fleet_lookup, rng)
            elif operator_choice < 0.8:
                candidate = move_swap(current, rng)
            else:
                candidate = move_couple_decouple(current, all_unit_ids, all_trip_ids, rng)

            cand_breakdown = compute_cost(candidate, trips_lookup, fleet_lookup, dist_lookup, maint_lookup)
            cand_cost = cand_breakdown["total"]
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
            "best_cost": round(best_cost, 1),
            "total_cost": round(best_cost, 1)
        })
        temp *= cooling_rate

    return best, best_cost, best_breakdown, history

# Main execution routine
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    inst = load_instance()
    fleet, trips, dist_lookup, maint_lookup = inst["fleet"], inst["trips"], inst["dist_lookup"], inst["maint_lookup"]
    fleet_lookup, trips_lookup = get_lookups(fleet, trips)

    # Initial greedy baseline evaluation using shared greedy_construction
    init_chains = greedy_construction(inst)
    init_breakdown = compute_cost(init_chains, trips_lookup, fleet_lookup, dist_lookup, maint_lookup)
    init_cost = init_breakdown["total"]
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

    # Save final schedule CSV
    schedule_df = chains_to_schedule_df(best_chains, trips)
    schedule_path = os.path.join(OUT_DIR, "final_schedule.csv")
    schedule_df.to_csv(schedule_path, index=False)

    # Save SA cost history CSV
    history_df = pd.DataFrame(history)
    history_path = os.path.join(OUT_DIR, "sa_cost_history.csv")
    history_df.to_csv(history_path, index=False)

    # Call plot_cost_history saving to cost_history.png
    plot_path = os.path.join(OUT_DIR, "cost_history.png")
    plot_cost_history(history, "Simulated Annealing", plot_path)

    # Save run summary JSON
    summary_path = os.path.join(OUT_DIR, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "initial_breakdown": init_breakdown,
            "final_breakdown": best_breakdown,
            "improvement_pct": round(improvement, 1),
            "hyperparameters": {
                "initial_temp": INITIAL_TEMP,
                "cooling_rate": COOLING_RATE,
                "min_temp": MIN_TEMP,
                "iterations_per_temp": ITERATIONS_PER_TEMP,
            },
            "algorithm": "simulated_annealing"
        }, f, indent=2)

    print(f"Saved: {schedule_path}")
    print(f"Saved: {history_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {summary_path}")

# Standalone execution entrypoint
if __name__ == "__main__":
    main()
