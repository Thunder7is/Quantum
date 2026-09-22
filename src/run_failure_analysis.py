# Failure analysis and comparison generator evaluating classical methods
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Directory paths for inputs and outputs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GREEDY_DIR = os.path.join(BASE_DIR, "outputs", "greedy")
SA_DIR = os.path.join(BASE_DIR, "outputs", "sa")
GA_DIR = os.path.join(BASE_DIR, "outputs", "ga")
COMP_DIR = os.path.join(BASE_DIR, "outputs", "comparison")

# Execute comprehensive failure comparison analysis and generate figures
def run_failure_analysis():
    os.makedirs(COMP_DIR, exist_ok=True)

    # Load summary artifacts from greedy and simulated annealing runs
    with open(os.path.join(GREEDY_DIR, "run_summary.json"), "r") as f:
        greedy_summary = json.load(f)

    with open(os.path.join(SA_DIR, "run_summary.json"), "r") as f:
        sa_summary = json.load(f)

    with open(os.path.join(GREEDY_DIR, "failure_analysis.json"), "r") as f:
        greedy_failures = json.load(f)

    greedy_bd = greedy_summary["breakdown"]
    sa_bd = sa_summary.get("final_breakdown", sa_summary.get("breakdown", {}))

    # 1. Generate failure_summary.json
    failure_summary = {
        "greedy": {
            "timing_violations": greedy_bd.get("timing_violations", 0),
            "uncovered_trips": len(greedy_failures.get("uncovered_trips", [])),
            "uncovered_demand_pax": greedy_bd.get("uncovered_demand", greedy_bd.get("uncovered_demand_passengers", 0)),
            "total_cost": greedy_bd.get("total", 0.0),
        },
        "sa": {
            "timing_violations": sa_bd.get("timing_violations", 0),
            "uncovered_trips": 0,
            "uncovered_demand_pax": sa_bd.get("uncovered_demand", sa_bd.get("uncovered_demand_passengers", 0)),
            "total_cost": sa_bd.get("total", 0.0),
        }
    }

    summary_path = os.path.join(COMP_DIR, "failure_summary.json")
    with open(summary_path, "w") as f:
        json.dump(failure_summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # 2. Generate classical_failure_chart.png
    metrics = ["timing_violations", "uncovered_demand_pax", "total_cost (scaled /100)", "units_used"]
    labels = ["Timing Violations", "Uncovered Demand (pax)", "Total Cost (/100)", "Units Used"]

    greedy_values = [
        float(greedy_bd.get("timing_violations", 0)),
        float(greedy_bd.get("uncovered_demand", greedy_bd.get("uncovered_demand_passengers", 0))),
        float(greedy_bd.get("total", 0.0)) / 100.0,
        float(greedy_bd.get("units_used", 0)),
    ]

    sa_values = [
        float(sa_bd.get("timing_violations", 0)),
        float(sa_bd.get("uncovered_demand", sa_bd.get("uncovered_demand_passengers", 0))),
        float(sa_bd.get("total", 0.0)) / 100.0,
        float(sa_bd.get("units_used", 0)),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(metrics))
    bar_width = 0.35

    bars1 = ax.bar(x - bar_width / 2, greedy_values, bar_width, label="Greedy Baseline", color="#d62728", edgecolor="black")
    bars2 = ax.bar(x + bar_width / 2, sa_values, bar_width, label="Simulated Annealing", color="#1f77b4", edgecolor="black")

    # Annotate numeric values above bars
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Feasibility threshold line at zero
    ax.axhline(0, color="red", linestyle="--", linewidth=1.8, label="Feasibility Threshold")

    ax.set_title("Where Classical Methods Fail — Rolling Stock Scheduling", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylabel("Metric Value / Scale", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()

    chart1_path = os.path.join(COMP_DIR, "classical_failure_chart.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {chart1_path}")

    # 3. Generate cost_reduction_chart.png
    sa_hist_path = os.path.join(SA_DIR, "sa_cost_history.csv")
    df_sa = pd.read_csv(sa_hist_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    steps = df_sa["step"]
    best_costs = df_sa["best_cost"]

    # Plot SA progression line
    ax.plot(steps, best_costs, color="#1f77b4", linewidth=2.2, label="SA Best Cost")
    if "current_cost" in df_sa.columns:
        ax.plot(steps, df_sa["current_cost"], color="#aec7e8", alpha=0.35, linewidth=1.0, label="SA Current Acceptance")

    init_cost = float(best_costs.iloc[0])
    final_cost = float(best_costs.iloc[-1])
    pct_improvement = 100.0 * (init_cost - final_cost) / init_cost if init_cost else 0.0

    # Annotate initial cost at start of search
    ax.annotate(f"Initial (Greedy): {init_cost:.1f}",
                xy=(steps.iloc[0], init_cost),
                xytext=(steps.iloc[0] + max(steps) * 0.05, init_cost + (max(best_costs) - min(best_costs)) * 0.08),
                arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
                fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="#d62728", facecolor="#ffebee"))

    # Annotate final cost and percentage improvement
    ax.annotate(f"Final (SA): {final_cost:.1f}\nImprovement: {pct_improvement:.1f}%",
                xy=(steps.iloc[-1], final_cost),
                xytext=(steps.iloc[-1] - max(steps) * 0.35, final_cost + (max(best_costs) - min(best_costs)) * 0.2),
                arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.5, headwidth=6),
                fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="#1f77b4", facecolor="#e8f4f8"))

    # Optional reference line from genetic algorithm best cost if available
    ga_summary_path = os.path.join(GA_DIR, "run_summary.json")
    if os.path.exists(ga_summary_path):
        with open(ga_summary_path, "r") as f:
            ga_summary = json.load(f)
        ga_cost = float(ga_summary.get("breakdown", {}).get("total", 0.0))
        if ga_cost > 0:
            ax.axhline(ga_cost, color="green", linestyle="--", linewidth=1.8, label=f"Optimal Reference (GA: {ga_cost:.1f})")

    ax.set_title("Simulated Annealing Cost Reduction Trajectory", fontsize=13, fontweight="bold")
    ax.set_xlabel("Annealing Steps (Iterations over Temperature)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Objective Cost", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()

    chart2_path = os.path.join(COMP_DIR, "cost_reduction_chart.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {chart2_path}")

    # 4. Trigger scaling failure simulation
    try:
        from scaling_failure_simulation import run_scaling_simulation
    except ImportError:
        from src.scaling_failure_simulation import run_scaling_simulation
    run_scaling_simulation()

# Standalone execution entrypoint
if __name__ == "__main__":
    run_failure_analysis()
