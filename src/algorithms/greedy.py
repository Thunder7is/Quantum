# Pure greedy dispatch heuristic solver for rolling stock circulation
import os
import json
import pandas as pd
from ..utils.data_loader import load_instance, get_lookups
from ..utils.cost import compute_cost
from ..utils.greedy_init import greedy_construction, chains_to_schedule_df

# Output directory path for greedy baseline
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "greedy"))

# Solve problem instance using greedy construction heuristic
def solve_greedy(fleet, trips, dist_lookup, maint_lookup):
    # Obtain lookup dictionary structures for fast evaluation
    fleet_lookup, trips_lookup = get_lookups(fleet, trips)
    # Construct circulation chains greedily
    chains, trip_assignment = greedy_construction(fleet, trips, dist_lookup)
    # Evaluate full cost breakdown
    breakdown = compute_cost(chains, trips, fleet, dist_lookup, maint_lookup)
    cost = breakdown["total"]
    return chains, cost, breakdown

# Standalone execution entrypoint
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    inst = load_instance()
    fleet, trips, dist_lookup, maint_lookup = inst["fleet"], inst["trips"], inst["dist_lookup"], inst["maint_lookup"]
    print(f"Loaded instance: {len(fleet)} units, {len(trips)} trips, {len(inst['stations'])} stations")

    # Run greedy solver
    chains, cost, breakdown = solve_greedy(fleet, trips, dist_lookup, maint_lookup)
    print("Greedy baseline solution cost breakdown:")
    print(json.dumps(breakdown, indent=2))

    # Save schedule dataframe
    schedule_df = chains_to_schedule_df(chains, trips)
    schedule_path = os.path.join(OUT_DIR, "final_schedule.csv")
    schedule_df.to_csv(schedule_path, index=False)

    # Save summary json
    summary_path = os.path.join(OUT_DIR, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"breakdown": breakdown, "method": "greedy"}, f, indent=2)

    print(f"Saved: {schedule_path}")
    print(f"Saved: {summary_path}")
