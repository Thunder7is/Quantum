# Scaling failure simulation analyzing all 4 algorithms across problem sizes
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
    from .algorithms.simulated_annealing import solve_sa
    from .algorithms.genetic_algorithm import solve_genetic_algorithm
    from .algorithms.nsga2 import solve_nsga2
except (ImportError, ValueError):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.utils.data_loader import load_instance, get_lookups
    from src.utils.cost import compute_cost
    from src.utils.greedy_init import greedy_construction
    from src.algorithms.simulated_annealing import solve_sa
    from src.algorithms.genetic_algorithm import solve_genetic_algorithm
    from src.algorithms.nsga2 import solve_nsga2

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

# Run scaling experiment across multiple trip counts for all 4 algorithms
def run_scaling_simulation():
    os.makedirs(OUT_DIR, exist_ok=True)
    inst = load_instance()
    fleet = inst["fleet"]
    base_trips = inst["trips"]
    dist_lookup = inst["dist_lookup"]
    maint_lookup = inst["maint_lookup"]

    trip_counts = [20, 40, 60, 80, 100, 120]
    all_records = []
    greedy_records = []

    print("=================================================================")
    print("  Running Scaling Benchmark Across All 4 Optimization Solvers")
    print(f"  Fleet: {len(fleet)} units | Trip Counts: {trip_counts}")
    print("=================================================================\n")

    for n in trip_counts:
        trips_scaled = build_scaled_trips(base_trips, n)
        fleet_lookup, trips_lookup = get_lookups(fleet, trips_scaled)
        inst_scaled = {"fleet": fleet, "trips": trips_scaled, "dist_lookup": dist_lookup, "maint_lookup": maint_lookup}

        print(f"--- Scaling Evaluation for n_trips = {n} ---")

        # 1. Evaluate Greedy Baseline
        t0 = time.time()
        chains_g = greedy_construction(inst_scaled)
        runtime_g = time.time() - t0
        bd_g = compute_cost(chains_g, trips_lookup, fleet_lookup, dist_lookup, maint_lookup)
        viols_g = bd_g.get("timing_violations", 0)
        uncovered_g = bd_g.get("uncovered_demand", bd_g.get("uncovered_demand_passengers", 0))
        cost_g = round(float(bd_g.get("total", 0.0)), 1)

        greedy_records.append({
            "n_trips": n,
            "timing_violations": viols_g,
            "uncovered_demand": uncovered_g,
            "runtime_sec": round(runtime_g, 4),
        })

        all_records.append({
            "n_trips": n,
            "algorithm": "Greedy",
            "timing_violations": viols_g,
            "uncovered_demand": uncovered_g,
            "total_cost": cost_g,
            "runtime_sec": round(runtime_g, 4),
        })
        print(f"  [Greedy] Violations: {viols_g:2d} | Uncovered: {uncovered_g:5d} | Cost: {cost_g:7.1f} | Runtime: {runtime_g:.4f}s")

        # 2. Evaluate Simulated Annealing
        t0 = time.time()
        _, cost_sa, bd_sa, _ = solve_sa(
            fleet, trips_scaled, dist_lookup, maint_lookup,
            initial_temp=500.0, cooling_rate=0.995, min_temp=0.5, iterations_per_temp=40
        )
        runtime_sa = time.time() - t0
        viols_sa = bd_sa.get("timing_violations", 0)
        uncovered_sa = bd_sa.get("uncovered_demand", bd_sa.get("uncovered_demand_passengers", 0))
        cost_sa = round(float(bd_sa.get("total", 0.0)), 1)

        all_records.append({
            "n_trips": n,
            "algorithm": "Simulated Annealing",
            "timing_violations": viols_sa,
            "uncovered_demand": uncovered_sa,
            "total_cost": cost_sa,
            "runtime_sec": round(runtime_sa, 4),
        })
        print(f"  [SA]     Violations: {viols_sa:2d} | Uncovered: {uncovered_sa:5d} | Cost: {cost_sa:7.1f} | Runtime: {runtime_sa:.4f}s")

        # 3. Evaluate Genetic Algorithm (pop_size=30, max_generations=50 for speed)
        t0 = time.time()
        _, cost_ga, bd_ga, _, _ = solve_genetic_algorithm(
            fleet, trips_scaled, dist_lookup, maint_lookup,
            pop_size=30, max_generations=50
        )
        runtime_ga = time.time() - t0
        viols_ga = bd_ga.get("timing_violations", 0)
        uncovered_ga = bd_ga.get("uncovered_demand", bd_ga.get("uncovered_demand_passengers", 0))
        cost_ga = round(float(bd_ga.get("total", 0.0)), 1)

        all_records.append({
            "n_trips": n,
            "algorithm": "Genetic Algorithm",
            "timing_violations": viols_ga,
            "uncovered_demand": uncovered_ga,
            "total_cost": cost_ga,
            "runtime_sec": round(runtime_ga, 4),
        })
        print(f"  [GA]     Violations: {viols_ga:2d} | Uncovered: {uncovered_ga:5d} | Cost: {cost_ga:7.1f} | Runtime: {runtime_ga:.4f}s")

        # 4. Evaluate NSGA-II (pop_size=30, generations=50 for speed)
        t0 = time.time()
        _, cost_nsga, bd_nsga, _, _ = solve_nsga2(
            fleet, trips_scaled, dist_lookup, maint_lookup,
            pop_size=30, generations=50
        )
        runtime_nsga = time.time() - t0
        viols_nsga = bd_nsga.get("timing_violations", 0)
        uncovered_nsga = bd_nsga.get("uncovered_demand", bd_nsga.get("uncovered_demand_passengers", 0))
        cost_nsga = round(float(bd_nsga.get("total", 0.0)), 1)

        all_records.append({
            "n_trips": n,
            "algorithm": "NSGA-II",
            "timing_violations": viols_nsga,
            "uncovered_demand": uncovered_nsga,
            "total_cost": cost_nsga,
            "runtime_sec": round(runtime_nsga, 4),
        })
        print(f"  [NSGA-II]Violations: {viols_nsga:2d} | Uncovered: {uncovered_nsga:5d} | Cost: {cost_nsga:7.1f} | Runtime: {runtime_nsga:.4f}s\n")

    # Save outputs/comparison/scaling_failure_all.csv
    df_all = pd.DataFrame(all_records)
    all_csv_path = os.path.join(OUT_DIR, "scaling_failure_all.csv")
    df_all.to_csv(all_csv_path, index=False)
    print(f"Saved master scaling CSV: {all_csv_path}")

    # Save existing single-algorithm greedy outputs
    df_greedy = pd.DataFrame(greedy_records)
    greedy_csv_path = os.path.join(OUT_DIR, "scaling_failure.csv")
    df_greedy.to_csv(greedy_csv_path, index=False)
    print(f"Saved greedy scaling CSV: {greedy_csv_path}")

    # Generate single-algorithm greedy plot
    fig_g, ax_g1 = plt.subplots(figsize=(8.5, 5))
    x_g = df_greedy["n_trips"]
    y_g_viols = df_greedy["timing_violations"]
    y_g_run = df_greedy["runtime_sec"]

    color_viol = "#d62728"
    ax_g1.set_xlabel("Number of Scheduled Timetable Trips (n_trips)", fontsize=11, fontweight="bold")
    ax_g1.set_ylabel("Timing Violations Count (Greedy)", color=color_viol, fontsize=11, fontweight="bold")
    line1 = ax_g1.plot(x_g, y_g_viols, color=color_viol, marker="o", linewidth=2.2, label="Timing Violations")
    ax_g1.tick_params(axis="y", labelcolor=color_viol)
    ax_g1.grid(True, linestyle="--", alpha=0.5)

    ax_g2 = ax_g1.twinx()
    color_run = "#1f77b4"
    ax_g2.set_ylabel("Runtime (seconds)", color=color_run, fontsize=11, fontweight="bold")
    line2 = ax_g2.plot(x_g, y_g_run, color=color_run, marker="s", linestyle="--", linewidth=1.8, label="Runtime (s)")
    ax_g2.tick_params(axis="y", labelcolor=color_run)

    viol_60 = df_greedy.loc[df_greedy["n_trips"] == 60, "timing_violations"].values[0]
    ax_g1.annotate(
        "SA improvement gap widens after 60 trips",
        xy=(60, viol_60),
        xytext=(62, viol_60 + max(2, max(y_g_viols) * 0.25)),
        arrowprops=dict(facecolor="#333333", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="#999999", facecolor="#fff9e6", alpha=0.9),
    )

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_g1.legend(lines, labels, loc="upper left", frameon=True)
    plt.title("Greedy Scaling Failure & Computational Profile", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    greedy_plot_path = os.path.join(OUT_DIR, "scaling_failure.png")
    plt.savefig(greedy_plot_path, dpi=300)
    plt.close(fig_g)
    print(f"Saved greedy scaling plot: {greedy_plot_path}")

    # Generate 2x2 subplot figure for all 4 algorithms
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    algo_styles = [
        {"name": "Greedy", "color": "#d62728", "marker": "o", "linestyle": "-"},
        {"name": "Simulated Annealing", "color": "#ff7f0e", "marker": "s", "linestyle": "-"},
        {"name": "Genetic Algorithm", "color": "#2ca02c", "marker": "^", "linestyle": "--"},
        {"name": "NSGA-II", "color": "#1f77b4", "marker": "D", "linestyle": "-."},
    ]

    metrics = [
        ("timing_violations", "Timing Violations Count", axes[0, 0]),
        ("uncovered_demand", "Uncovered Passenger Demand", axes[0, 1]),
        ("total_cost", "Total Operational Cost", axes[1, 0]),
        ("runtime_sec", "Runtime (seconds)", axes[1, 1]),
    ]

    for metric_col, ylabel, ax in metrics:
        for style in algo_styles:
            algo_name = style["name"]
            sub_df = df_all[df_all["algorithm"] == algo_name]
            ax.plot(
                sub_df["n_trips"],
                sub_df[metric_col],
                label=algo_name,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                markersize=6,
            )

        # Vertical dashed line at x=60 labeled Current instance size
        ax.axvline(x=60, color="#555555", linestyle=":", linewidth=1.8, label="Current instance size")
        ax.set_xlabel("Number of Scheduled Trips (n_trips)", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="best", frameon=True)

    fig.suptitle("Algorithm Degradation Under Scaling — Rolling Stock Scheduling", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()

    all_plot_path = os.path.join(OUT_DIR, "scaling_all_algorithms.png")
    plt.savefig(all_plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved all algorithms scaling plot: {all_plot_path}")

# Standalone execution entrypoint
if __name__ == "__main__":
    run_scaling_simulation()
