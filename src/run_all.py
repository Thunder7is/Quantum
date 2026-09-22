# Master execution benchmark running all four rolling stock scheduling algorithms
import os
import sys
import time
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Support execution as direct script or package module
try:
    from .algorithms.greedy import main as run_greedy
    from .algorithms.simulated_annealing import main as run_sa
    from .algorithms.genetic_algorithm import main as run_ga
    from .algorithms.nsga2 import main as run_nsga2
except (ImportError, ValueError):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.algorithms.greedy import main as run_greedy
    from src.algorithms.simulated_annealing import main as run_sa
    from src.algorithms.genetic_algorithm import main as run_ga
    from src.algorithms.nsga2 import main as run_nsga2

# Base directories for inputs and comparison outputs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
COMP_DIR = os.path.join(OUTPUTS_DIR, "comparison")

# Execute all four solvers in sequence, record timings, and compile comparative reports
def main():
    os.makedirs(COMP_DIR, exist_ok=True)
    runtimes = {}

    print("=================================================================")
    print("  Master Rolling Stock Benchmark: Running All 4 Solvers")
    print("=================================================================\n")

    # Step 1: Run each algorithm in sequence and record individual execution runtimes
    print("[1/4] Executing Pure Greedy Baseline...")
    t0 = time.time()
    run_greedy()
    runtimes["Greedy"] = round(time.time() - t0, 2)

    print("\n[2/4] Executing Simulated Annealing Solver...")
    t0 = time.time()
    run_sa()
    runtimes["Simulated Annealing"] = round(time.time() - t0, 2)

    print("\n[3/4] Executing Genetic Algorithm Solver...")
    t0 = time.time()
    run_ga()
    runtimes["Genetic Algorithm"] = round(time.time() - t0, 2)

    print("\n[4/4] Executing NSGA-II Multi-Objective Solver...")
    t0 = time.time()
    run_nsga2()
    runtimes["NSGA-II"] = round(time.time() - t0, 2)

    # Step 2: Load run_summary.json artifacts from each algorithm output folder
    with open(os.path.join(OUTPUTS_DIR, "greedy", "run_summary.json"), "r") as f:
        greedy_sum = json.load(f)

    with open(os.path.join(OUTPUTS_DIR, "sa", "run_summary.json"), "r") as f:
        sa_sum = json.load(f)

    with open(os.path.join(OUTPUTS_DIR, "ga", "run_summary.json"), "r") as f:
        ga_sum = json.load(f)

    with open(os.path.join(OUTPUTS_DIR, "nsga2", "run_summary.json"), "r") as f:
        nsga2_sum = json.load(f)

    # Extract standardized metric breakdowns
    breakdowns = {
        "Greedy": greedy_sum["breakdown"],
        "Simulated Annealing": sa_sum.get("final_breakdown", sa_sum.get("breakdown")),
        "Genetic Algorithm": ga_sum["breakdown"],
        "NSGA-II": nsga2_sum.get("best_compromise_breakdown", nsga2_sum.get("breakdown")),
    }

    # Step 3: Build master comparison records
    rows = []
    for algo, bd in breakdowns.items():
        rows.append({
            "algorithm": algo,
            "total_cost": round(float(bd.get("total", bd.get("total_cost", 0.0))), 1),
            "deadhead_km": round(float(bd.get("deadhead_km", bd.get("total_deadhead_km", 0.0))), 1),
            "units_used": int(bd.get("units_used", 0)),
            "maintenance_violation_km": round(float(bd.get("maintenance_violation_km", 0.0))),
            "uncovered_demand": int(bd.get("uncovered_demand", bd.get("uncovered_demand_passengers", 0))),
            "timing_violations": int(bd.get("timing_violations", 0)),
            "runtime_sec": runtimes[algo],
        })

    # Save outputs/comparison/master_comparison.csv
    df_comparison = pd.DataFrame(rows)
    csv_path = os.path.join(COMP_DIR, "master_comparison.csv")
    df_comparison.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Step 4: Generate 4-panel figure master_comparison.png
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    metrics_info = [
        ("total_cost", "Total Objective Cost (lower = better)", axes[0, 0]),
        ("deadhead_km", "Deadhead Transit (km)", axes[0, 1]),
        ("units_used", "Activated Fleet Units (count)", axes[1, 0]),
        ("uncovered_demand", "Uncovered Passenger Demand", axes[1, 1]),
    ]

    # Prescribed color palette: red for Greedy, orange for SA, green for GA, blue for NSGA-II
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    algos = df_comparison["algorithm"].tolist()

    for metric_col, title, ax in metrics_info:
        vals = df_comparison[metric_col].tolist()
        bars = ax.bar(algos, vals, color=colors, width=0.55, edgecolor="#222222", linewidth=1.0)
        # Annotate bar values
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}" if isinstance(h, float) else f"{h}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.tick_params(axis="x", rotation=15, labelsize=9.5)

    plt.suptitle("Master Benchmark: Rolling Stock Scheduling Algorithm Comparison", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    img_path = os.path.join(COMP_DIR, "master_comparison.png")
    plt.savefig(img_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {img_path}")

    # Step 5: Print ranked summary table to stdout (ranked by total_cost ascending)
    df_ranked = df_comparison.sort_values("total_cost").reset_index(drop=True)
    df_ranked.insert(0, "rank", range(1, len(df_ranked) + 1))

    print("\n=================================================================")
    print("  Ranked Algorithm Performance Summary (Ranked by Total Cost)")
    print("=================================================================")
    print(df_ranked.to_string(index=False))
    print("=================================================================\n")

# Standalone execution entrypoint
if __name__ == "__main__":
    main()
