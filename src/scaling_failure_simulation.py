# Scaling failure simulation analyzing classical greedy heuristic performance across problem sizes
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Support execution as direct script or package module
try:
    from .utils.data_loader import load_instance, get_lookups
    from .utils.cost import compute_cost
    from .utils.greedy_init import greedy_construction
except (ImportError, ValueError):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.utils.data_loader import load_instance, get_lookups
    from src.utils.cost import compute_cost
    from src.utils.greedy_init import greedy_construction

# Directory path for scaling comparison artifacts
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "comparison"))

# Synthesize scaled problem instance with specified number of trips
def build_scaled_trips(base_trips: pd.DataFrame, n_trips: int) -> pd.DataFrame:
    base_n = len(base_trips)
    if n_trips <= base_n:
        # Take chronological slice of scheduled timetable trips
        return base_trips.iloc[:n_trips].copy().reset_index(drop=True)

    # Replicate trips with staggered departure windows to simulate network congestion
    scaled_records = base_trips.to_dict("records")
    extra_needed = n_trips - base_n

    for idx in range(extra_needed):
        template = dict(base_trips.iloc[idx % base_n])
        duration = template["arrival_min"] - template["departure_min"]
        # Shift departure to peak congestion windows while keeping operational interval
        new_dep = (template["departure_min"] + 25 * ((idx // base_n) + 1)) % (22 * 60)
        new_dep = max(4 * 60, new_dep)
        template["trip_id"] = f"T{base_n + idx + 1:03d}"
        template["departure_min"] = new_dep
        template["arrival_min"] = new_dep + duration
        scaled_records.append(template)

    scaled_df = pd.DataFrame(scaled_records)
    return scaled_df.sort_values("departure_min").reset_index(drop=True)

# Run scaling experiment across multiple trip counts
def run_scaling_simulation():
    os.makedirs(OUT_DIR, exist_ok=True)
    large_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "large"))
    if os.path.exists(os.path.join(large_dir, "trips.csv")):
        inst = load_instance(large_dir)
    else:
        inst = load_instance()
    fleet = inst["fleet"]
    base_trips = inst["trips"]
    dist_lookup = inst["dist_lookup"]
    maint_lookup = inst["maint_lookup"]

    trip_counts = [20, 40, 60, 80, 100, 120]
    records = []

    print("=================================================================")
    print("  Running Scaling Failure Simulation (Greedy Heuristic)")
    print(f"  Fleet: {len(fleet)} units | Trip Counts: {trip_counts}")
    print("=================================================================\n")

    for n in trip_counts:
        trips_scaled = build_scaled_trips(base_trips, n)
        fleet_lookup, trips_lookup = get_lookups(fleet, trips_scaled)

        # Time greedy construction execution
        t0 = time.time()
        inst_scaled = {"fleet": fleet, "trips": trips_scaled, "dist_lookup": dist_lookup, "maint_lookup": maint_lookup}
        chains = greedy_construction(inst_scaled)
        runtime = time.time() - t0

        # Evaluate performance and constraint violations
        bd = compute_cost(chains, trips_lookup, fleet_lookup, dist_lookup, maint_lookup)
        timing_viols = bd.get("timing_violations", 0)
        uncovered = bd.get("uncovered_demand", bd.get("uncovered_demand_passengers", 0))

        records.append({
            "n_trips": n,
            "timing_violations": timing_viols,
            "uncovered_demand": uncovered,
            "runtime_sec": round(runtime, 4),
        })

        print(f"  Trips: {n:3d} | Timing Violations: {timing_viols:2d} | Uncovered Demand: {uncovered:5d} | Runtime: {runtime:.4f}s")

    # Save results to outputs/comparison/scaling_failure.csv
    df_results = pd.DataFrame(records)
    csv_path = os.path.join(OUT_DIR, "scaling_failure.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved scaling CSV: {csv_path}")

    # Generate dual-axis scaling plot
    fig, ax1 = plt.subplots(figsize=(8.5, 5))

    x = df_results["n_trips"]
    y_violations = df_results["timing_violations"]
    y_runtime = df_results["runtime_sec"]

    # Primary axis for timing violations
    color_viol = "#d62728"
    ax1.set_xlabel("Number of Scheduled Timetable Trips (n_trips)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Timing Violations Count (Greedy)", color=color_viol, fontsize=11, fontweight="bold")
    line1 = ax1.plot(x, y_violations, color=color_viol, marker="o", linewidth=2.2, label="Timing Violations")
    ax1.tick_params(axis="y", labelcolor=color_viol)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Secondary axis for execution runtime
    ax2 = ax1.twinx()
    color_run = "#1f77b4"
    ax2.set_ylabel("Runtime (seconds)", color=color_run, fontsize=11, fontweight="bold")
    line2 = ax2.plot(x, y_runtime, color=color_run, marker="s", linestyle="--", linewidth=1.8, label="Runtime (s)")
    ax2.tick_params(axis="y", labelcolor=color_run)

    # Add research annotation at x=60
    viol_60 = df_results.loc[df_results["n_trips"] == 60, "timing_violations"].values[0]
    ax1.annotate(
        "SA improvement gap widens after 60 trips",
        xy=(60, viol_60),
        xytext=(62, viol_60 + max(2, max(y_violations) * 0.25)),
        arrowprops=dict(facecolor="#333333", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="#999999", facecolor="#fff9e6", alpha=0.9),
    )

    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True)

    plt.title("Greedy Scaling Failure & Computational Profile", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    # Save scaling plot
    plot_path = os.path.join(OUT_DIR, "scaling_failure.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved scaling plot: {plot_path}")

# Standalone execution entrypoint
if __name__ == "__main__":
    run_scaling_simulation()
