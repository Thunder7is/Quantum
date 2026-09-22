# Pure greedy dispatch heuristic baseline solver with failure analysis
import os
import sys
import json
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

# Output directory path for greedy baseline artifacts
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "greedy"))

# Perform detailed failure diagnostics on circulation schedule
def analyze_failures(chains: dict, fleet_lookup: dict, trips_lookup: dict, dist_lookup: dict) -> dict:
    # Map each trip to its assigned rolling stock units
    trip_to_units = {tid: [] for tid in trips_lookup}
    for uid, chain in chains.items():
        for tid in chain:
            trip_to_units[tid].append(uid)

    timing_violations = set()
    uncovered_trips = []
    overloaded_trips = []

    # Check timing feasibility sequentially along each unit chain
    for uid, chain in chains.items():
        if not chain:
            continue
        unit = fleet_lookup[uid]
        location = unit["home_depot"]
        avail_time = unit["available_from_min"]
        sorted_chain = sorted(chain, key=lambda tid: trips_lookup[tid]["departure_min"])
        for tid in sorted_chain:
            trip = trips_lookup[tid]
            travel_time, _ = travel(dist_lookup, location, trip["origin_station"])
            if avail_time + travel_time > trip["departure_min"]:
                timing_violations.add(tid)
            location = trip["destination_station"]
            avail_time = trip["arrival_min"] + trip["min_turnaround_min"]

    # Evaluate passenger demand fulfillment and identify unserved trips
    for tid, uids in trip_to_units.items():
        trip = trips_lookup[tid]
        demand = trip.get("expected_passenger_demand", 0)
        supplied_capacity = sum(fleet_lookup[u]["capacity"] for u in uids)
        if supplied_capacity == 0:
            uncovered_trips.append(tid)
        elif supplied_capacity < demand:
            overloaded_trips.append(tid)

    return {
        "timing_violations": sorted(list(timing_violations)),
        "uncovered_trips": sorted(uncovered_trips),
        "overloaded_trips": sorted(overloaded_trips),
    }

# Solve problem instance using greedy construction heuristic directly
def solve_greedy(fleet, trips, dist_lookup, maint_lookup):
    fleet_lookup, trips_lookup = get_lookups(fleet, trips)
    inst = {"fleet": fleet, "trips": trips, "dist_lookup": dist_lookup, "maint_lookup": maint_lookup}
    # Direct greedy construction without iterative improvement
    chains = greedy_construction(inst)
    # Evaluate full cost breakdown
    breakdown = compute_cost(chains, trips_lookup, fleet_lookup, dist_lookup, maint_lookup)
    cost = breakdown["total"]
    return chains, cost, breakdown

# Main execution routine
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    inst = load_instance()
    fleet, trips, dist_lookup, maint_lookup = inst["fleet"], inst["trips"], inst["dist_lookup"], inst["maint_lookup"]
    fleet_lookup, trips_lookup = get_lookups(fleet, trips)

    print("=================================================================")
    print("  Greedy Baseline Solver (Classical Baseline 2)")
    print(f"  Fleet: {len(fleet)} units | Trips: {len(trips)} | Stations: {len(inst['stations'])}")
    print("=================================================================\n")

    # Run greedy heuristic solver
    chains, cost, breakdown = solve_greedy(fleet, trips, dist_lookup, maint_lookup)
    print("Greedy baseline solution cost breakdown:")
    print(json.dumps(breakdown, indent=2))

    # Perform failure diagnostic analysis
    failures = analyze_failures(chains, fleet_lookup, trips_lookup, dist_lookup)
    print("\n--- Failure Analysis ---")
    print(f"Timing violations count: {len(failures['timing_violations'])}")
    print(f"Timing violated trips: {failures['timing_violations']}")
    print(f"Uncovered trips count: {len(failures['uncovered_trips'])}")
    print(f"Uncovered trips: {failures['uncovered_trips']}")
    print(f"Overloaded trips count (capacity < demand): {len(failures['overloaded_trips'])}")
    print(f"Overloaded trips: {failures['overloaded_trips']}")

    # Save final schedule CSV
    schedule_df = chains_to_schedule_df(chains, trips)
    schedule_path = os.path.join(OUT_DIR, "final_schedule.csv")
    schedule_df.to_csv(schedule_path, index=False)

    # Save cost history PNG as a single point flat line
    history = [{"step": 0, "total_cost": cost, "best_cost": cost}]
    plot_path = os.path.join(OUT_DIR, "cost_history.png")
    plot_cost_history(history, "Greedy Baseline", plot_path)

    # Save run summary JSON
    summary_path = os.path.join(OUT_DIR, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"breakdown": breakdown, "method": "greedy"}, f, indent=2)

    # Save failure analysis JSON
    failure_path = os.path.join(OUT_DIR, "failure_analysis.json")
    with open(failure_path, "w") as f:
        json.dump(failures, f, indent=2)

    print(f"\nSaved schedule: {schedule_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved failure analysis: {failure_path}")

# Standalone execution entrypoint
if __name__ == "__main__":
    main()
